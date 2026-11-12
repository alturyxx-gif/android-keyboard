package org.futo.inputmethod.latin.uix

import org.futo.inputmethod.latin.uix.voicepower.PowerMath
import org.futo.inputmethod.latin.uix.voicepower.PowerMath.Sample
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PowerMathTest {
    private fun samples(currentRaw: Long, voltageMv: Int = 4000, n: Int = 10) =
        (0 until n).map { Sample(it * 100L, currentRaw, voltageMv) }

    @Test
    fun detectsMilliampsVsMicroamps() {
        assertEquals(1, PowerMath.currentScaleToMicroamps(samples(-450_000)))   // µA, negative = discharging
        assertEquals(1000, PowerMath.currentScaleToMicroamps(samples(450)))     // mA
        assertEquals(1000, PowerMath.currentScaleToMicroamps(samples(-450)))
    }

    @Test
    fun powerFromCurrentAndVoltage() {
        // 500 mA at 4.0 V = 2000 mW, whatever the unit/sign convention
        assertEquals(2000.0, PowerMath.powerMw(Sample(0, -500_000, 4000), 1), 1e-6)
        assertEquals(2000.0, PowerMath.powerMw(Sample(0, 500, 4000), 1000), 1e-6)
    }

    @Test
    fun energyPerRunSubtractsIdle() {
        val scale = 1L
        val idle = PowerMath.phaseStats(samples(-300_000), scale)   // 1200 mW
        val active = PowerMath.phaseStats(samples(-800_000), scale) // 3200 mW
        // 2000 mW extra for 20 s over 40 runs = 1000 mJ per run
        val r = PowerMath.engineResult(active, idle, idle, runs = 40, seconds = 20.0)
        assertEquals(1200.0, r.idlePowerMw, 1e-6)
        assertEquals(1000.0, r.energyPerRunMj, 1e-6)
        assertEquals(0.5, r.secondsPerRun, 1e-9)
    }

    @Test
    fun noRunsMeansNoEnergy() {
        val s = PowerMath.phaseStats(samples(-300_000), 1)
        assertEquals(0.0, PowerMath.engineResult(s, s, s, 0, 20.0).energyPerRunMj, 0.0)
    }

    @Test
    fun combineRoundsWeightsByRuns() {
        val a = PowerMath.EngineResult(10, 10.0, 3000.0, 1000.0, 100.0, 1.0)
        val b = PowerMath.EngineResult(30, 10.0, 3000.0, 1000.0, 200.0, 0.33)
        val c = PowerMath.combine(listOf(a, b))!!
        assertEquals(40, c.runs)
        assertEquals(175.0, c.energyPerRunMj, 1e-9)
        assertNull(PowerMath.combine(emptyList()))
    }

    @Test
    fun verdictWording() {
        val cpu = PowerMath.EngineResult(10, 10.0, 0.0, 0.0, 1000.0, 1.0)
        val gpu = PowerMath.EngineResult(10, 10.0, 0.0, 0.0, 600.0, 1.0)
        assertTrue(PowerMath.verdict(cpu, gpu, 11.0).startsWith("GPU used about 40% less"))
        assertTrue(PowerMath.verdict(gpu, cpu, 11.0).contains("MORE"))
        assertTrue(PowerMath.verdict(cpu, cpu, 11.0).contains("about the same"))
        assertTrue(PowerMath.verdict(null, cpu, 11.0).startsWith("Not enough"))
    }

    @Test
    fun transcriptAgreementDetectsBrokenGpuOutput() {
        val cpu = "And so my fellow Americans ask not what your country can do for you"
        assertTrue(PowerMath.transcriptAgreement(cpu, "And so, my fellow Americans, ask not what your country can do for you.") > 0.95)
        // The garbage the GPU produced with Whisper tiny on the S25 Ultra
        assertTrue(PowerMath.transcriptAgreement(cpu, "!....London, oh") < PowerMath.MIN_AGREEMENT)
        assertTrue(PowerMath.transcriptAgreement(cpu, "! Heynumber of arms, thank booze apart correctionsanuts") < PowerMath.MIN_AGREEMENT)
        assertEquals(1.0, PowerMath.transcriptAgreement("", ""), 0.0)
    }
}
