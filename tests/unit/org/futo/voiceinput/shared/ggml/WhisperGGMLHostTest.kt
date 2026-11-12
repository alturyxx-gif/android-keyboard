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
