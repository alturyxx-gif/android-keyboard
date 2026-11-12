package org.futo.inputmethod.latin.uix.voicepower

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.futo.inputmethod.latin.uix.ResourceHelper
import org.futo.inputmethod.latin.uix.VOICE_USE_GPU
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.voiceinput.shared.ggml.DecodingMode
import org.futo.voiceinput.shared.ggml.VoiceEngineSettings
import org.futo.voiceinput.shared.ggml.WhisperGGML
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import java.util.Locale

/**
 * Voice power test (custom build): measures how much battery energy one transcription costs
 * on the CPU vs. the GPU, using the phone's own battery current/voltage readings.
 *
 * Sequence (twice): idle 10 s -> CPU transcribing in a loop 20 s -> idle 10 s -> GPU 20 s,
 * then a final idle. Each engine's extra power above the surrounding idle periods, times the
 * time, divided by the number of transcriptions = energy per transcription.
 */
class VoicePowerTest(private val context: Context) {
    class TestError(message: String) : Exception(message)

    companion object {
        const val IDLE_SECONDS = 10
        const val ACTIVE_SECONDS = 20
        const val ROUNDS = 2
        const val SAMPLE_INTERVAL_MS = 100L
        val TOTAL_SECONDS = ROUNDS * (2 * IDLE_SECONDS + 2 * ACTIVE_SECONDS) + IDLE_SECONDS + 15

        fun reportFile(context: Context) = File(context.noBackupFilesDir, "voice_power_report.txt")
    }

    private val battery = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    private fun voltageMv(): Int {
        val intent: Intent? = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return intent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
    }

    private fun isCharging(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && battery.isCharging) return true
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val plugged = intent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        return plugged != 0
    }

    private fun loadAudio(): FloatArray {
        val bytes = context.assets.open("voice_power_test.wav").use { it.readBytes() }
        var pos = 12
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4)
            val size = ByteBuffer.wrap(bytes, pos + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
            if (id == "data") {
                val bb = ByteBuffer.wrap(bytes, pos + 8, size).order(ByteOrder.LITTLE_ENDIAN)
                return FloatArray(size / 2) { bb.short / 32768f }
            }
            pos += 8 + size + (size and 1)
        }
        throw TestError("Test audio is missing")
    }

    private enum class Phase { Idle, Cpu, Gpu }
    private data class Tagged(val phaseIndex: Int, val sample: PowerMath.Sample)

    /**
     * Runs the test. [progress] gets a status line and a 0..1 fraction. Returns the report text,
     * which is also saved to [reportFile].
     */
    suspend fun run(progress: (String, Float) -> Unit): String = withContext(Dispatchers.Default) {
        if (isCharging()) throw TestError("Unplug the phone first: while charging, the battery readings can't show how much power voice input uses.")
        val firstCurrent = battery.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        if (firstCurrent == 0L || firstCurrent == Long.MIN_VALUE) throw TestError("This phone doesn't report battery current, so power can't be measured.")

        val model = ResourceHelper.tryFindingVoiceInputModelForLocale(context, Locale.ENGLISH)
            ?: throw TestError("No English voice model found.")
        val audio = loadAudio()
        val audioSeconds = audio.size / 16000.0

        // Same engine setup as the keyboard's voice input
        VoiceEngineSettings.gpuCrashMarker = File(context.noBackupFilesDir, "voice_gpu_attempt")
        VoiceEngineSettings.gpuFailedMarker = File(context.noBackupFilesDir, "voice_gpu_failed")
        VoiceEngineSettings.diagnosticsFile = File(context.noBackupFilesDir, "voice_last_run.txt")
        VoiceEngineSettings.nativeLibDir = context.applicationInfo.nativeLibraryDir ?: ""
        VoiceEngineSettings.gpuLogFile = File(context.noBackupFilesDir, "voice_gpu_log.txt")

        progress("Loading the model on the CPU…", 0f)
        VoiceEngineSettings.useGpu = false
        val cpu = model.loadGGML(context)
        progress("Loading the model on the GPU (may freeze briefly)…", 0.02f)
        VoiceEngineSettings.useGpu = true
        val gpu = try { model.loadGGML(context) } finally {
            VoiceEngineSettings.useGpu = context.getSetting(VOICE_USE_GPU)
        }

        try {
            if (!gpu.isUsingGpu) throw TestError("The GPU isn't available for voice input on this phone (it fell back to the CPU), so there is nothing to compare. Check \"Last voice input run\" in Voice Input settings.")

            suspend fun transcribe(w: WhisperGGML): String =
                w.infer(audio, "", arrayOf("en"), arrayOf(), DecodingMode.BeamSearch5, false, {})

            progress("Warming up both engines…", 0.04f)
            val cpuText = transcribe(cpu)
            val gpuText = transcribe(gpu)
            if (gpuText.isBlank()) throw TestError("The GPU returned no text, so it can't be measured.")

            val samples = Collections.synchronizedList(ArrayList<Tagged>())
            val phaseIndex = java.util.concurrent.atomic.AtomicInteger(0)
            val phases = ArrayList<Pair<Phase, Int>>()    // phase type, runs completed
            val durations = ArrayList<Double>()
            val start = System.currentTimeMillis()

            coroutineScope {
                val sampler = launch(Dispatchers.IO) {
                    var v = voltageMv()
                    var n = 0
                    while (isActive) {
                        if (n++ % 10 == 0) v = voltageMv()
                        val c = battery.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
                        samples.add(Tagged(phaseIndex.get(), PowerMath.Sample(System.currentTimeMillis(), c, v)))
                        delay(SAMPLE_INTERVAL_MS)
                    }
                }

                suspend fun phase(type: Phase, seconds: Int, label: String) {
                    val t0 = System.currentTimeMillis()
                    var runs = 0
                    while (System.currentTimeMillis() - t0 < seconds * 1000L) {
                        val elapsed = (System.currentTimeMillis() - start) / 1000f
                        progress(label, (0.05f + 0.95f * elapsed / TOTAL_SECONDS).coerceAtMost(0.99f))
                        when (type) {
                            Phase.Idle -> delay(250)
                            Phase.Cpu -> { transcribe(cpu); runs++ }
                            Phase.Gpu -> { transcribe(gpu); runs++ }
                        }
                    }
                    durations.add((System.currentTimeMillis() - t0) / 1000.0)
                    phases.add(type to runs)
                    phaseIndex.incrementAndGet()
                }

                for (round in 1..ROUNDS) {
                    phase(Phase.Idle, IDLE_SECONDS, "Round $round/$ROUNDS: measuring idle power…")
                    phase(Phase.Cpu, ACTIVE_SECONDS, "Round $round/$ROUNDS: CPU transcribing…")
                    phase(Phase.Idle, IDLE_SECONDS, "Round $round/$ROUNDS: measuring idle power…")
                    phase(Phase.Gpu, ACTIVE_SECONDS, "Round $round/$ROUNDS: GPU transcribing…")
                }
                phase(Phase.Idle, IDLE_SECONDS, "Final idle measurement…")
                sampler.cancel()
            }

            // ---- Analysis ----
            val all = synchronized(samples) { ArrayList(samples) }
            val scale = PowerMath.currentScaleToMicroamps(all.map { it.sample })
            val stats = phases.indices.map { i -> PowerMath.phaseStats(all.filter { it.phaseIndex == i }.map { it.sample }, scale) }

            val cpuRounds = ArrayList<PowerMath.EngineResult>()
            val gpuRounds = ArrayList<PowerMath.EngineResult>()
            for (i in phases.indices) {
                val (type, runs) = phases[i]
                if (type == Phase.Idle) continue
                val r = PowerMath.engineResult(stats[i], stats[i - 1], stats[i + 1], runs, durations[i])
                if (type == Phase.Cpu) cpuRounds.add(r) else gpuRounds.add(r)
            }
            val cpuR = PowerMath.combine(cpuRounds)
            val gpuR = PowerMath.combine(gpuRounds)

            val distinct = all.map { it.sample.currentRaw }.distinct().size
            val report = buildString {
                append("Voice power test — ${Build.MANUFACTURER} ${Build.MODEL}\n")
                append("Model: ${model.key(context)}\n")
                append("Test audio: %.1f s of speech, transcribed repeatedly\n\n".format(audioSeconds))
                append(PowerMath.verdict(cpuR, gpuR, audioSeconds)).append("\n\n")
                fun line(name: String, r: PowerMath.EngineResult?) {
                    if (r == null) { append("$name: no data\n"); return }
                    append("$name: ${r.runs} transcriptions, %.2f s each\n".format(r.secondsPerRun))
                    append("   power %.0f mW while running vs %.0f mW idle → %.0f mJ per transcription\n".format(r.avgPowerMw, r.idlePowerMw, r.energyPerRunMj))
                }
                line("CPU", cpuR)
                line("GPU", gpuR)
                append("\nRounds (energy per transcription):\n")
                cpuRounds.forEachIndexed { i, r -> append("   round ${i + 1}: CPU %.0f mJ, GPU %.0f mJ\n".format(r.energyPerRunMj, gpuRounds.getOrNull(i)?.energyPerRunMj ?: 0.0)) }
                append("\nMeasurement quality: ${all.size} readings, $distinct distinct current values, ")
                append(if (scale == 1000L) "phone reports mA" else "phone reports µA").append("\n")
                if (distinct < all.size / 10) append("Note: the phone updates its current reading slowly, so treat small differences (<15%) as noise.\n")
                append("CPU said: \"${cpuText.take(80)}\"\nGPU said: \"${gpuText.take(80)}\"\n")
            }
            try { reportFile(context).writeText(report) } catch (_: Exception) { }
            progress("Done", 1f)
            report
        } finally {
            cpu.close()
            gpu.close()
        }
    }
}
