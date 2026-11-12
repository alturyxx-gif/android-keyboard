package org.futo.voiceinput.shared.ggml

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Runs the keyboard's real voice engine class (WhisperGGML -> libvoiceinput -> transcribe.cpp)
 * on this computer, against FUTO's built-in English model and a real speech recording.
 *
 * Needs the desktop build of libvoiceinput (see voiceinput-shared/src/main/cpp/CMakeLists.txt);
 * pass its folder via the VOICE_HOST_LIB_DIR environment variable. Skipped otherwise.
 */
class WhisperGGMLHostTest {
    private val libDir = System.getenv("VOICE_HOST_LIB_DIR") ?: ""
    private val root = File(System.getProperty("user.dir"))
    private val model = File(root, "voiceinput-shared/src/main/ml/tiny_en_acft_q8_0.bin.not.tflite")
    private val audio = File(root, "voiceinput-shared/src/main/cpp/transcribe.cpp/samples/jfk.wav")

    private fun available() = libDir.isNotEmpty() && File(libDir, "libvoiceinput.so").exists() && model.exists() && audio.exists()

    /** Reads a 16 kHz mono 16-bit PCM WAV into floats. */
    private fun readWav(f: File): FloatArray {
        val bytes = f.readBytes()
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
        error("no data chunk")
    }

    private fun transcribe(w: WhisperGGML, context: String = ""): String = runBlocking {
        w.infer(readWav(audio), "", arrayOf("en"), arrayOf(), DecodingMode.BeamSearch5, false, {}, context)
    }

    @Test
    fun transcribesRealSpeechWithBuiltInModel() {
        assumeTrue("host voice library not built", available())
        VoiceEngineSettings.useGpu = false
        val w = WhisperGGML(model)
        val text = transcribe(w)
        println("Transcript: $text")
        assertTrue(text.lowercase().contains("ask not what your country can do for you"))
        runBlocking { w.close() }
    }

    @Test
    fun textContextIsAcceptedAndDoesNotBreakTranscription() {
        assumeTrue("host voice library not built", available())
        VoiceEngineSettings.useGpu = false
        val w = WhisperGGML(model)
        val text = transcribe(w, context = "And so, my fellow Americans:")
        println("Transcript with context: $text")
        assertTrue(text.lowercase().contains("your country"))
        runBlocking { w.close() }
    }

    @Test
    fun gpuRequestFallsBackToCpuAndClearsCrashMarker() {
        assumeTrue("host voice library not built", available())
        val marker = File.createTempFile("voice_gpu", ".marker").apply { delete() }
        VoiceEngineSettings.useGpu = true
        VoiceEngineSettings.gpuCrashMarker = marker
        VoiceEngineSettings.gpuDisabledAfterCrash = false
        val w = WhisperGGML(model)
        // This build has no GPU backend, so it must have fallen back to the CPU...
        assertFalse(w.isUsingGpu)
        // ...and must not leave a "GPU crashed" marker behind
        assertFalse(marker.exists())
        assertTrue(transcribe(w).lowercase().contains("country"))
        runBlocking { w.close() }
        VoiceEngineSettings.gpuCrashMarker = null
    }

    @Test
    fun previousGpuCrashForcesCpu() {
        assumeTrue("host voice library not built", available())
        val marker = File.createTempFile("voice_gpu", ".marker").apply { writeText("gpu") }
        VoiceEngineSettings.useGpu = true
        VoiceEngineSettings.gpuCrashMarker = marker
        VoiceEngineSettings.gpuDisabledAfterCrash = false
        val w = WhisperGGML(model)
        assertTrue(VoiceEngineSettings.gpuDisabledAfterCrash)
        assertFalse(w.isUsingGpu)
        runBlocking { w.close() }
        VoiceEngineSettings.gpuCrashMarker = null
        marker.delete()
    }

    /**
     * Stage 1.2: the real Whisper large-v3-turbo GGUF (as imported into the keyboard).
     * Pass its path via VOICE_TURBO_MODEL. Skipped otherwise.
     */
    @Test
    fun transcribesWithWhisperLargeV3Turbo() {
        val turboPath = System.getenv("VOICE_TURBO_MODEL") ?: ""
        assumeTrue("turbo model not provided", available() && turboPath.isNotEmpty() && File(turboPath).exists())
        VoiceEngineSettings.useGpu = false
        // The import screen must classify it as a voice model
        val arch = File(turboPath).inputStream().use { org.futo.inputmethod.latin.uix.GgufInfo.readArchitecture(it) }
        assertEquals("whisper", arch)
        assertFalse(org.futo.inputmethod.latin.uix.GgufInfo.isTextModel(arch))

        val w = WhisperGGML(File(turboPath))
        val start = System.nanoTime()
        val text = transcribe(w, context = "My fellow Americans,")
        val secs = (System.nanoTime() - start) / 1e9
        println("Turbo transcript (%.1f s on this machine's CPU): %s".format(secs, text))
        assertTrue(text.lowercase().contains("ask not what your country can do for you"))
        runBlocking { w.close() }
    }

    /** Optional: NVIDIA Parakeet TDT 0.6B v2 GGUF via VOICE_PARAKEET_MODEL (imported like turbo). */
    @Test
    fun transcribesWithParakeet() {
        val path = System.getenv("VOICE_PARAKEET_MODEL") ?: ""
        assumeTrue("parakeet model not provided", available() && path.isNotEmpty() && File(path).exists())
        VoiceEngineSettings.useGpu = false
        val arch = File(path).inputStream().use { org.futo.inputmethod.latin.uix.GgufInfo.readArchitecture(it) }
        assertFalse(org.futo.inputmethod.latin.uix.GgufInfo.isTextModel(arch))
        val w = WhisperGGML(File(path))
        val start = System.nanoTime()
        val text = transcribe(w, context = "This is ignored by Parakeet")
        val secs = (System.nanoTime() - start) / 1e9
        println("Parakeet [$arch] transcript (%.1f s on this machine's CPU): %s".format(secs, text))
        assertTrue(text.lowercase().contains("ask not what your country can do for you"))
        runBlocking { w.close() }
    }

    /** Timing breakdown for a short (3 s) clip: model load vs. transcription. */
    @Test
    fun timingLoadVersusTranscribeShortClip() {
        val paths = listOfNotNull(System.getenv("VOICE_PARAKEET_MODEL"), System.getenv("VOICE_TURBO_MODEL"), model.path)
        assumeTrue(available())
        VoiceEngineSettings.useGpu = false
        val clip = readWav(audio).copyOfRange(0, 16000 * 3)
        for (p in paths) {
            if (!File(p).exists()) continue
            val t0 = System.nanoTime()
            val w = WhisperGGML(File(p))
            val t1 = System.nanoTime()
            val text = runBlocking { w.infer(clip, "", arrayOf("en"), arrayOf(), DecodingMode.BeamSearch5, false, {}) }
            val t2 = System.nanoTime()
            val text2 = runBlocking { w.infer(clip, "", arrayOf("en"), arrayOf(), DecodingMode.BeamSearch5, false, {}) }
            val t3 = System.nanoTime()
            println("TIMING %s: load %.2f s, first run %.2f s, second run %.2f s -> %s".format(
                File(p).name, (t1 - t0) / 1e9, (t2 - t1) / 1e9, (t3 - t2) / 1e9, text2))
            runBlocking { w.close() }
        }
    }

    /** GPU is only allowed for Parakeet-type models; decided from the file header. */
    @Test
    fun gpuOnlyAllowedForParakeetModels() {
        assumeTrue(available())
        val diag = File.createTempFile("voice_diag", ".txt")
        VoiceEngineSettings.diagnosticsFile = diag
        VoiceEngineSettings.gpuCrashMarker = null
        VoiceEngineSettings.gpuFailedMarker = null
        VoiceEngineSettings.useGpu = true

        fun noteFor(path: File): String {
            val w = WhisperGGML(path)
            transcribe(w)
            runBlocking { w.close() }
            return diag.readText()
        }

        // FUTO's legacy Whisper .bin: never on the GPU
        assertTrue(noteFor(model).contains("GPU not used for Whisper models"))

        val turbo = System.getenv("VOICE_TURBO_MODEL")?.let { File(it) }
        if (turbo != null && turbo.exists()) {
            assertTrue(noteFor(turbo).contains("GPU not used for whisper models"))
        }

        // Parakeet is allowed: on this machine there's no GPU, so it reports that instead
        val parakeet = System.getenv("VOICE_PARAKEET_MODEL")?.let { File(it) }
        if (parakeet != null && parakeet.exists()) {
            val note = noteFor(parakeet)
            assertFalse(note.contains("GPU not used for"))
        }
        VoiceEngineSettings.useGpu = false
        VoiceEngineSettings.diagnosticsFile = null
        diag.delete()
    }

    @Test
    fun invalidModelThrows() {
        assumeTrue("host voice library not built", available())
        VoiceEngineSettings.useGpu = false
        val bad = File.createTempFile("bad", ".bin").apply { writeText("not a model") }
        var threw = false
        try { WhisperGGML(bad) } catch (e: InvalidModelException) { threw = true }
        assertEquals(true, threw)
        bad.delete()
    }
}
