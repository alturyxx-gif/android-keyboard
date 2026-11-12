package org.futo.voiceinput.shared.ggml

import android.util.Log
import androidx.annotation.Keep
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.withContext
import java.io.File

@OptIn(DelicateCoroutinesApi::class)
val inferenceContext = newSingleThreadContext("whisper-ggml-inference")

enum class DecodingMode(val value: Int) {
    Greedy(0),
    BeamSearch5(5)
}

class BailLanguageException(val language: String): Exception()
class InferenceCancelledException : Exception()
class InvalidModelException : Exception("The voice model could not be loaded")

/**
 * Engine settings set by the keyboard (custom build).
 */
object VoiceEngineSettings {
    /** Try the GPU (Vulkan) first, with automatic CPU fallback. */
    @Volatile var useGpu: Boolean = true

    /**
     * Marker file used to detect a GPU attempt that crashed the process (a native driver crash
     * can't be caught). If the marker still exists on the next start, the GPU is skipped.
     */
    @Volatile var gpuCrashMarker: File? = null

    /** Set when a crash was detected so the UI can tell the user. */
    @Volatile var gpuDisabledAfterCrash: Boolean = false
}

/**
 * Speech recognition engine. Custom build: backed by transcribe.cpp (the engine used by the
 * newer standalone FUTO Voice Input app) in its own native library, instead of the old
 * whisper.cpp copy inside the keyboard. Supports Whisper (incl. large-v3-turbo), Parakeet,
 * Nemotron, Moonshine and FUTO's asr4all GGUF models, plus the old FUTO .bin Whisper models.
 */
@Keep
class WhisperGGML(
    file: File
) {
    companion object {
        init {
            System.loadLibrary("voiceinput")
        }
    }

    private var handle: Long = 0L
    var isUsingGpu: Boolean = false
        private set

    init {
        val marker = VoiceEngineSettings.gpuCrashMarker
        var tryGpu = VoiceEngineSettings.useGpu
        if (tryGpu && marker != null && marker.exists()) {
            Log.w("WhisperGGML", "Previous GPU attempt crashed, using CPU")
            VoiceEngineSettings.gpuDisabledAfterCrash = true
            tryGpu = false
        }

        if (tryGpu) try { marker?.writeText("gpu") } catch (_: Exception) { }
        handle = openNative(file.absolutePath, tryGpu)
        if (handle != 0L) isUsingGpu = isUsingGpuNative(handle)
        if (!isUsingGpu) try { marker?.delete() } catch (_: Exception) { }

        if(handle == 0L) {
            throw InvalidModelException()
        }
    }

    private var partialResultCallback: (String) -> Unit = { }

    @Keep
    private fun invokePartialResult(text: String) {
        partialResultCallback(text.trim())
    }

    // empty languages = autodetect any language
    // 1 language = will force that language
    // 2 or more languages = autodetect between those languages
    @Throws(BailLanguageException::class, InferenceCancelledException::class)
    suspend fun infer(
        samples: FloatArray,
        prompt: String,
        languages: Array<String>,
        bailLanguages: Array<String>,
        decodingMode: DecodingMode,
        suppressNonSpeechTokens: Boolean,
        partialResultCallback: (String) -> Unit,
        textContext: String = "",
    ): String = withContext(inferenceContext) {
        if(handle == 0L) {
            throw IllegalStateException("WhisperGGML has already been closed, cannot infer")
        }
        this@WhisperGGML.partialResultCallback = partialResultCallback

        val result = (inferNative(handle, samples, prompt, textContext, languages, bailLanguages, decodingMode.value, suppressNonSpeechTokens) ?: "").trim()

        // A GPU inference completed without crashing: the GPU works on this phone
        if (isUsingGpu) try { VoiceEngineSettings.gpuCrashMarker?.delete() } catch (_: Exception) { }

        if(result.contains("<>CANCELLED<>")) {
            if(result.contains("flag")) {
                throw InferenceCancelledException()
            } else if(result.contains("lang=")) {
                val language = result.split("lang=")[1]
                throw BailLanguageException(language)
            } else {
                throw IllegalStateException("Cancelled for unknown reason")
            }

        } else {
            return@withContext result
        }
    }

    fun cancel() {
        if(handle == 0L) return
        cancelNative(handle)
    }

    suspend fun close() = withContext(inferenceContext) {
        if(handle != 0L) {
            closeNative(handle)
        }
        handle = 0L
    }

    private external fun openNative(path: String, useGpu: Boolean): Long
    private external fun inferNative(handle: Long, samples: FloatArray, prompt: String, textContext: String, languages: Array<String>, bailLanguages: Array<String>, decodingMode: Int, suppressNonSpeechTokens: Boolean): String?
    private external fun isUsingGpuNative(handle: Long): Boolean
    private external fun cancelNative(handle: Long)
    private external fun closeNative(handle: Long)
}
