package org.futo.inputmethod.latin.tap

import android.content.Context
import org.futo.inputmethod.keyboard.Key
import org.futo.inputmethod.keyboard.Keyboard

/**
 * Applies the context-aware decision (Tap plan, Stage 4) on top of FUTO's key detection.
 * Called by KeyDetector for every hit test; must be fast and must never make things worse
 * when it has nothing to go on.
 */
object TapKeyResolver {
    private var cachedKeyboard: Keyboard? = null
    private var letterKeys: List<LetterKey> = emptyList()
    private var keyByLetter: Map<Char, Key> = emptyMap()
    @Volatile private var appContext: Context? = null

    fun attach(context: Context) {
        appContext = context.applicationContext
        TapPredictor.refreshSettings(context)
    }

    fun lettersOf(keyboard: Keyboard): List<LetterKey> = synchronized(this) {
        if (keyboard !== cachedKeyboard) {
            val keys = keyboard.sortedKeys.filter { Character.isLetter(it.code) }
                .distinctBy { Character.toLowerCase(it.code) }
            letterKeys = keys.map {
                LetterKey(Character.toLowerCase(it.code).toChar(), it.x.toFloat(), it.y.toFloat(),
                    it.width.toFloat(), it.height.toFloat())
            }
            keyByLetter = keys.associateBy { Character.toLowerCase(it.code).toChar() }
            cachedKeyboard = keyboard
        }
        letterKeys
    }

    /**
     * [baseKey] is what FUTO's detection chose. Returns the key to use (usually the same).
     */
    @JvmStatic
    fun resolve(keyboard: Keyboard?, x: Int, y: Int, baseKey: Key?): Key? {
        if (keyboard == null || baseKey == null || !TapPredictor.enabled) return baseKey
        if (!Character.isLetter(baseKey.code)) return baseKey
        val snap = TapPredictor.current() ?: return baseKey
        val context = appContext ?: return baseKey
        val keys = lettersOf(keyboard)
        if (keys.size < 10) return baseKey
        // Only for keyboards made of plain lower/upper-case letters keys (not symbols pages)
        val params = TapDecision.Params(strength = TapPredictor.strength * TapKeyFeedback.strengthMultiplier())
        val r = TapDecision.contextRule(keys, x.toFloat(), y.toFloat(), snap.probs, TapPredictor.touchModel(context), params)
            ?: return baseKey
        val chosen = keyByLetter[r.letter] ?: return baseKey
        if (Character.toLowerCase(chosen.code) != Character.toLowerCase(baseKey.code)) {
            // Only count it as "changed" when it differs from the key under the finger
            if (r.changed) TapKeyFeedback.onTapChanged()
            return chosen
        }
        return baseKey
    }
}
