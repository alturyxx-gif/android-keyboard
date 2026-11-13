package org.futo.inputmethod.latin.swipe

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.datastore.preferences.core.booleanPreferencesKey
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.ml.inference.SwipeDecoder
import java.io.File
import java.util.zip.ZipInputStream

/** "Use my custom swipe model" (Stage 6 of the swipe plan). */
val UseCustomSwipeModel = SettingsKey(booleanPreferencesKey("swipe_use_custom_model"), false)

/**
 * A swipe encoder trained with swipe-training/ on your PC, imported as the zip that export.py
 * writes (model_fp32.pte + metadata.json). It replaces FUTO's encoder; FUTO's layout refiner
 * only works with FUTO's encoder, so it's switched off for custom models (the context model
 * and dictionary stay).
 */
object CustomSwipeModel {
    private const val TAG = "CustomSwipeModel"
    private val json = Json { ignoreUnknownKeys = true }

    data class Info(val displayName: String, val codename: String, val sizeBytes: Long, val config: String)

    private fun dir(context: Context) = File(context.filesDir, "swipe-models/custom")
    fun encoderFile(context: Context) = File(dir(context), "model_fp32.pte")

    fun isInstalled(context: Context) =
        encoderFile(context).isFile && File(dir(context), "metadata.json").isFile

    /** Path to use for the swipe encoder right now, or null to use FUTO's. */
    fun activePath(context: Context): String? =
        if (context.getSetting(UseCustomSwipeModel) && isInstalled(context)) encoderFile(context).absolutePath else null

    fun info(context: Context): Info? {
        val meta = File(dir(context), "metadata.json").takeIf { it.isFile } ?: return null
        return try {
            val o = json.parseToJsonElement(meta.readText()).jsonObject
            val cfg = o["model_config"]?.jsonObject
            Info(
                displayName = o["display_name"]?.jsonPrimitive?.content ?: "custom",
                codename = o["codename"]?.jsonPrimitive?.content ?: "",
                sizeBytes = encoderFile(context).length(),
                config = cfg?.let { "dim ${it["dim"]}, ${it["blocks"]} blocks, ${it["attn_layers"]} attention layers" } ?: "",
            )
        } catch (e: Exception) {
            null
        }
    }

    /** Checks metadata.json describes an encoder with the inputs/outputs the engine expects. */
    fun checkMetadata(text: String): String? = try {
        val o: JsonObject = json.parseToJsonElement(text).jsonObject
        when {
            o["kind"]?.jsonPrimitive?.content != "encoder" -> "metadata.json is not for a swipe encoder"
            o["max_keys"]?.jsonPrimitive?.int != 64 -> "model must take 64 key slots"
            o["input_shape"]?.jsonArray?.map { it.jsonPrimitive.int } != listOf(1, 2, 64) -> "model must take a 64-point path"
            else -> null
        }
    } catch (e: Exception) {
        "metadata.json can't be read: ${e.message}"
    }

    /**
     * Imports a model zip from [uri]. The model is test-loaded in the real swipe engine (which
     * runs it once) before it replaces the current custom model. Returns an error message, or
     * null on success.
     */
    fun import(context: Context, uri: Uri): String? {
        val staging = File(context.filesDir, "swipe-models/staging").apply { deleteRecursively(); mkdirs() }
        try {
            var gotPte = false
            var gotMeta = false
            context.contentResolver.openInputStream(uri)?.use { input ->
                ZipInputStream(input).use { zip ->
                    while (true) {
                        val e = zip.nextEntry ?: break
                        val name = e.name.substringAfterLast('/')
                        if (name != "model_fp32.pte" && name != "metadata.json") continue
                        File(staging, name).outputStream().use { zip.copyTo(it) }
                        if (name == "model_fp32.pte") gotPte = true else gotMeta = true
                    }
                }
            } ?: return "Couldn't open the file"
            if (!gotPte || !gotMeta) return "The zip must contain model_fp32.pte and metadata.json (made by export.py)"
            checkMetadata(File(staging, "metadata.json").readText())?.let { return it }

            // Test-load in the real engine: it runs the model once and fails on unsupported ops
            try {
                SwipeDecoder(encoderPath = File(staging, "model_fp32.pte").absolutePath, beamWidth = 32, useExpansion = false).close()
            } catch (e: Throwable) {
                return "The swipe engine couldn't load this model: ${e.message}"
            }

            val target = dir(context)
            target.deleteRecursively()
            target.parentFile?.mkdirs()
            if (!staging.renameTo(target)) return "Couldn't save the model"
            SwipeModelState.changed()
            return null
        } catch (e: Exception) {
            Log.e(TAG, "Import failed", e)
            return "Import failed: ${e.message}"
        } finally {
            staging.deleteRecursively()
        }
    }

    fun delete(context: Context) {
        dir(context).deleteRecursively()
        SwipeModelState.changed()
    }
}

/** Bumped whenever the custom model or the choice changes, so the keyboard reloads the engine. */
object SwipeModelState {
    @Volatile var generation = 0
        private set

    fun changed() { generation++ }
}
