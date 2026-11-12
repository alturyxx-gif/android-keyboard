package org.futo.inputmethod.latin.uix.voicepower

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Pure maths for the voice power test (custom build). No Android dependencies, unit tested.
 *
 * The phone reports battery current (BatteryManager CURRENT_NOW) and voltage. Power while a
 * phase runs = current x voltage. The extra energy one transcription costs is the phase's
 * average power minus the idle power measured right around it, times the phase duration,
 * divided by the number of transcriptions done in that phase.
 */
object PowerMath {
    data class Sample(val timeMs: Long, val currentRaw: Long, val voltageMv: Int)

    /**
     * BatteryManager.CURRENT_NOW is specified in microamps, but some phones report milliamps,
     * and the sign convention varies. A phone drawing power typically uses 100-3000 mA, so if
     * the median magnitude is below 20000 the values are taken to be milliamps.
     */
    fun currentScaleToMicroamps(samples: List<Sample>): Long {
        if (samples.isEmpty()) return 1
        val mags = samples.map { abs(it.currentRaw) }.sorted()
        val median = mags[mags.size / 2]
        return if (median in 1 until 20_000) 1000 else 1
    }

    /** Power in milliwatts for one sample. */
    fun powerMw(s: Sample, scale: Long): Double = abs(s.currentRaw) * scale / 1000.0 * s.voltageMv / 1000.0

    data class PhaseStats(val avgPowerMw: Double, val stdDevMw: Double, val samples: Int, val distinctReadings: Int)

    fun phaseStats(samples: List<Sample>, scale: Long): PhaseStats {
        if (samples.isEmpty()) return PhaseStats(0.0, 0.0, 0, 0)
        val p = samples.map { powerMw(it, scale) }
        val mean = p.average()
        val sd = sqrt(p.sumOf { (it - mean) * (it - mean) } / p.size)
        return PhaseStats(mean, sd, samples.size, samples.map { it.currentRaw }.distinct().size)
    }

    data class EngineResult(
        val runs: Int,
        val seconds: Double,
        val avgPowerMw: Double,
        val idlePowerMw: Double,
        /** Extra energy per transcription above idle, in millijoules. */
        val energyPerRunMj: Double,
        val secondsPerRun: Double,
    )

    fun engineResult(active: PhaseStats, idleBefore: PhaseStats, idleAfter: PhaseStats, runs: Int, seconds: Double): EngineResult {
        val idle = listOf(idleBefore, idleAfter).filter { it.samples > 0 }.map { it.avgPowerMw }.average().let { if (it.isNaN()) 0.0 else it }
        val extra = (active.avgPowerMw - idle).coerceAtLeast(0.0)
        val energy = if (runs > 0) extra * seconds / runs else 0.0
        return EngineResult(runs, seconds, active.avgPowerMw, idle, energy, if (runs > 0) seconds / runs else 0.0)
    }

    /** Combines several rounds of the same engine (weighted by number of runs). */
    fun combine(results: List<EngineResult>): EngineResult? {
        val r = results.filter { it.runs > 0 }
        if (r.isEmpty()) return null
        val runs = r.sumOf { it.runs }
        val secs = r.sumOf { it.seconds }
        return EngineResult(
            runs = runs,
            seconds = secs,
            avgPowerMw = r.sumOf { it.avgPowerMw * it.seconds } / secs,
            idlePowerMw = r.sumOf { it.idlePowerMw * it.seconds } / secs,
            energyPerRunMj = r.sumOf { it.energyPerRunMj * it.runs } / runs,
            secondsPerRun = secs / runs,
        )
    }

    /** One-line verdict comparing GPU with CPU. */
    fun verdict(cpu: EngineResult?, gpu: EngineResult?, audioSeconds: Double): String {
        if (cpu == null || gpu == null) return "Not enough data to compare."
        if (cpu.energyPerRunMj <= 0.0 || gpu.energyPerRunMj <= 0.0)
            return "The battery readings were too coarse to separate the engines from idle. Try again with the screen brightness fixed and nothing else running."
        val ratio = gpu.energyPerRunMj / cpu.energyPerRunMj
        val pct = ((1 - ratio) * 100).toInt()
        val per10 = { e: EngineResult -> e.energyPerRunMj / audioSeconds * 10 }
        val energyLine = when {
            ratio < 0.9 -> "GPU used about $pct% less energy per transcription than the CPU."
            ratio > 1.1 -> "GPU used about ${-pct}% MORE energy per transcription than the CPU."
            else -> "GPU and CPU used about the same energy per transcription (within 10%)."
        }
        return energyLine + " Per 10 s of speech: CPU %.0f mJ, GPU %.0f mJ (a phone battery holds ~70,000,000 mJ).".format(per10(cpu), per10(gpu))
    }
}
