package org.futo.inputmethod.latin.xlm.personal

import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import org.futo.inputmethod.latin.uix.SettingsKey
import java.io.File

/** How strongly the personalisation adapter is applied (0 = off, 1 = full). */
val PersonalisationStrength = SettingsKey(
    floatPreferencesKey("personalisation_adapter_strength"),
    0.75f
)

/** Threads used by the typing model (Stage 2.8). FUTO used 1. */
val TypingModelThreads = SettingsKey(
    intPreferencesKey("typing_model_threads"),
    2
)

/**
 * File layout for the personalisation adapter (Stage 2.2), next to the base model:
 *   <model>.gguf            base model, never modified
 *   <model>.lora            active adapter
 *   <model>.lora.prev       previous adapter (for "undo last training")
 *   <adapter>.scale         optional strength multiplier (0..1) chosen by the quality check
 */
object PersonalAdapterFiles {
    fun adapterFor(model: File): File = File(model.parentFile, model.nameWithoutExtension + ".lora")
    fun previousFor(model: File): File = File(model.parentFile, model.nameWithoutExtension + ".lora.prev")
    fun pendingFor(model: File): File = File(model.parentFile, model.nameWithoutExtension + ".lora.new")

    fun activeAdapter(model: File): File? = adapterFor(model).takeIf { it.isFile && it.length() > 0 }

    private fun scaleFileFor(adapter: File) = File(adapter.path + ".scale")

    /**
     * Strength multiplier for [adapter] (1.0 unless the quality check found the full strength
     * hurt everyday English and kept the adapter at a lower strength).
     */
    fun scaleMultiplier(adapter: File): Float = try {
        scaleFileFor(adapter).takeIf { it.isFile }?.readText()?.trim()?.toFloatOrNull()?.coerceIn(0f, 1f) ?: 1f
    } catch (e: Exception) { 1f }

    fun setScaleMultiplier(adapter: File, multiplier: Float) {
        val f = scaleFileFor(adapter)
        if (multiplier >= 1f) f.delete() else f.writeText(multiplier.toString())
    }

    private fun move(from: File, to: File): Boolean {
        scaleFileFor(to).delete()
        if (scaleFileFor(from).isFile) scaleFileFor(from).renameTo(scaleFileFor(to))
        return from.renameTo(to)
    }

    /** Makes a freshly trained adapter active, keeping the old one as the rollback point. */
    fun promote(model: File): Boolean {
        val pending = pendingFor(model)
        if (!pending.isFile) return false
        val active = adapterFor(model)
        val prev = previousFor(model)
        if (active.isFile) {
            prev.delete()
            move(active, prev)
        }
        return move(pending, active)
    }

    /** Goes back to the adapter from before the last training run. */
    fun rollBack(model: File): Boolean {
        val prev = previousFor(model)
        val active = adapterFor(model)
        if (!prev.isFile) {
            // No earlier adapter: rolling back means no personalisation
            scaleFileFor(active).delete()
            return active.delete()
        }
        active.delete()
        return move(prev, active)
    }

    fun deleteAll(model: File) {
        listOf(adapterFor(model), previousFor(model), pendingFor(model)).forEach {
            it.delete()
            scaleFileFor(it).delete()
        }
    }

    fun hasPrevious(model: File) = previousFor(model).isFile
}
