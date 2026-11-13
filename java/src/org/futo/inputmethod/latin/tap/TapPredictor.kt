package org.futo.inputmethod.latin.tap

import android.content.Context
import android.util.Log
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.floatPreferencesKey
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.inputmethod.latin.xlm.LanguageModel
import org.futo.inputmethod.latin.xlm.LanguageModelScope
import org.futo.inputmethod.latin.xlm.grammar.RomanUrdu
import org.futo.inputmethod.latin.xlm.personal.PersonalisationStore
import java.io.File

/** Context-aware key detection on/off (Tap plan, Stage 4). Needs "Smart key-hit detection" on. */
val TapContextAware = SettingsKey(booleanPreferencesKey("tap_context_aware"), true)
/** How strongly likely letters pull taps near key edges (0 = touch only, 1 = strong). */
val TapStrength = SettingsKey(floatPreferencesKey("tap_strength"), 0.5f)
/** Draw the invisible key areas on the keyboard (for testing). */
val TapDebugOverlay = SettingsKey(booleanPreferencesKey("tap_debug_overlay"), false)

/**
 * Next-letter likelihoods for the next tap (Tap plan, Stage 3), worked out in the background
 * right after each key press, plus your touch model (Stage 2).
 *
 * Key detection asks [current]; it only uses a result computed for the latest key press
 * (otherwise FUTO's behaviour applies to that tap), so typing never waits.
 */
object TapPredictor {
    private const val TAG = "TapPredictor"

    data class Snapshot(val generation: Int, val prefix: String, val probs: FloatArray?)

    @Volatile private var generation = 0
    @Volatile private var snapshot: Snapshot? = null
    @Volatile var enabled = false
        private set
    @Volatile var strength = 0.5f
        private set
    @Volatile var debugOverlay = false
        private set
    /** Bumped when something the debug overlay draws has changed */
    @Volatile var overlayVersion = 0
        private set

    /** Set by the IME: the typing model if it's loaded and for the current language */
    var modelProvider: (() -> LanguageModel?)? = null
    /** Set by the IME: next letters the dictionaries allow after a prefix (for replay testing) */
    var dictionaryProvider: ((String) -> Set<Int>?)? = null

    private val scope = CoroutineScope(SupervisorJob() + LanguageModelScope)

    // ---- touch model ----
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; allowStructuredMapKeys = true }
    private var touch: TouchModel? = null
    private var learnedSinceSave = 0
    private fun touchFile(context: Context) = File(context.filesDir, "tap-touch-model.json")

    fun touchModel(context: Context): TouchModel = synchronized(this) {
        touch ?: (try {
            touchFile(context).takeIf { it.isFile }?.readText()?.let { json.decodeFromString(TouchModel.serializer(), it) }
        } catch (e: Exception) { null } ?: TouchModel()).also { touch = it }
    }

    fun resetTouchModel(context: Context) = synchronized(this) {
        touch = TouchModel()
        touchFile(context).delete()
        overlayVersion++
    }

    private fun saveTouch(context: Context) {
        val text = synchronized(this) {
            learnedSinceSave = 0
            json.encodeToString(TouchModel.serializer(), touch ?: return)
        }
        val app = context.applicationContext
        Thread { try { touchFile(app).writeText(text) } catch (_: Exception) { } }
            .apply { priority = Thread.MIN_PRIORITY }.start()
    }

    /** Learn where you tap from a word whose intended letters are known. */
    fun learnTouch(context: Context, intended: String, tapX: IntArray, tapY: IntArray, keys: KeyBoxes, typed: String) {
        val tm = touchModel(context)
        var n = 0
        synchronized(this) {
            for (i in intended.indices) {
                if (i >= tapX.size || i >= typed.length) break
                // Only positions where the letter that went in is the one meant (typos are
                // learned from mid-word fixes instead)
                if (!intended[i].equals(typed[i], ignoreCase = true)) continue
                val k = keys.index(intended[i])
                if (k < 0) continue
                val (cx, cy) = keys.centre(k)
                tm.learn(intended[i], (tapX[i] - cx) / keys.w[k], (tapY[i] - cy) / keys.h[k])
                n++
            }
            learnedSinceSave += n
        }
        if (learnedSinceSave >= 200) saveTouch(context)
        if (n > 0) overlayVersion++
    }

    fun learnFix(context: Context, intended: Char, x: Int, y: Int, keys: KeyBoxes) {
        val tm = touchModel(context)
        val k = keys.index(intended)
        if (k < 0) return
        val (cx, cy) = keys.centre(k)
        synchronized(this) { tm.learn(intended, (x - cx) / keys.w[k], (y - cy) / keys.h[k]); learnedSinceSave++ }
    }

    // ---- next-letter likelihoods ----

    fun refreshSettings(context: Context) {
        enabled = context.getSetting(TapContextAware)
        strength = context.getSetting(TapStrength)
        debugOverlay = context.getSetting(TapDebugOverlay)
        overlayVersion++
    }

    /** The result for the latest key press, or null (use FUTO's behaviour). */
    @JvmStatic
    fun current(): Snapshot? = snapshot?.takeIf { it.generation == generation }

    /**
     * Called after every key event. [prefix]: letters of the word being typed ("" at the start
     * of a word). [dictValid]: next letters the dictionary allows (null if unknown).
     */
    @JvmStatic
    fun onInput(context: Context, prefix: String, prevWords: String?, dictValid: Set<Int>?,
                afterBackspace: Boolean, isPrivate: Boolean, boostingOn: Boolean) {
        val gen = ++generation
        snapshot = null
        overlayVersion++
        if (!enabled || !boostingOn || afterBackspace || isPrivate) return
        val words = prevWords?.split(' ')?.filter { it.isNotBlank() && it != "<S>" }?.takeLast(12) ?: emptyList()
        val sentence = words.joinToString(" ")
        val valid = dictValid?.mapNotNull { cp -> Character.toLowerCase(cp).toChar().takeIf { it in 'a'..'z' } }?.toSet()
        val urdu = RomanUrdu.isMostlyRomanUrdu("$sentence $prefix")
        scope.launch {
            try {
                val lm = if (urdu) null else modelProvider?.invoke()?.nextLetterProbsLocked(sentence, prefix)
                val personal = personalCompletions(prefix, urdu)
                val probs = LetterMix.combine(lm, valid, personal, prefix)
                if (generation == gen) {
                    snapshot = Snapshot(gen, prefix, probs)
                    overlayVersion++
                }
            } catch (e: Exception) {
                Log.w(TAG, "Next-letter prediction failed", e)
            }
        }
    }

    /** Synchronous version for replay testing (runs on the typing model's thread). */
    suspend fun computeForReplay(sentence: String, prefix: String, valid: Set<Char>?): FloatArray? {
        val urdu = RomanUrdu.isMostlyRomanUrdu("$sentence $prefix")
        val lm = if (urdu) null else modelProvider?.invoke()?.nextLetterProbs(sentence, prefix)
        return LetterMix.combine(lm, valid, personalCompletions(prefix, urdu), prefix)
    }

    private fun personalCompletions(prefix: String, urdu: Boolean): List<Pair<String, Int>> = try {
        if (prefix.isEmpty()) emptyList()
        else if (urdu) PersonalisationStore.urduVocab.completions(prefix, 8).map { it to 2 }
        else PersonalisationStore.memory.completions(prefix, 8, minCount = 2)
    } catch (e: Exception) { emptyList() }
}

/** Combining the sources of next-letter likelihood (pure, unit tested). */
object LetterMix {
    /** Weight of letters the dictionary says can't continue any word */
    const val INVALID_FACTOR = 0.25f
    /** Share of probability given to letters continuing words you often type */
    const val PERSONAL_SHARE = 0.25f

    fun combine(lm: FloatArray?, dictValid: Set<Char>?, personal: List<Pair<String, Int>>, prefix: String): FloatArray? {
        val p = FloatArray(26)
        when {
            lm != null && lm.size == 26 -> lm.copyInto(p)
            !dictValid.isNullOrEmpty() -> dictValid.forEach { c -> if (c in 'a'..'z') p[c - 'a'] = 1f }
            personal.isEmpty() -> return null
            else -> p.fill(1f)
        }
        if (lm != null && !dictValid.isNullOrEmpty()) {
            for (i in 0 until 26) if (('a' + i) !in dictValid) p[i] *= INVALID_FACTOR
        }
        normalize(p)
        if (personal.isNotEmpty()) {
            val bonus = FloatArray(26)
            val total = personal.sumOf { it.second }.toFloat()
            for ((w, c) in personal) {
                if (w.length <= prefix.length) continue
                val l = w[prefix.length].lowercaseChar()
                if (l in 'a'..'z') bonus[l - 'a'] += c / total
            }
            if (bonus.sum() > 0f) {
                normalize(bonus)
                for (i in 0 until 26) p[i] = (1 - PERSONAL_SHARE) * p[i] + PERSONAL_SHARE * bonus[i]
            }
        }
        return p
    }

    private fun normalize(p: FloatArray) {
        val s = p.sum()
        if (s > 0f) for (i in p.indices) p[i] /= s
    }
}
