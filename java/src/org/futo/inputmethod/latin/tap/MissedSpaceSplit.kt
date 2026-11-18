package org.futo.inputmethod.latin.tap

import org.futo.inputmethod.keyboard.Keyboard

/**
 * Two words typed as one because a letter above the space bar was hit instead of space
 * ("hellobthere" -> "hello there"). FUTO's dictionary only corrects this when the tap was
 * right next to the space bar; this works wherever on the key the tap landed.
 */
object MissedSpaceSplit {
    /** Letters right above the space bar on the current layout (default: QWERTY's c v b n). */
    fun splitLetters(keyboard: Keyboard?): Set<Char> {
        if (keyboard == null) return DEFAULT
        val space = TapKeyResolver.spaceOf(keyboard) ?: return DEFAULT
        val keys = TapKeyResolver.lettersOf(keyboard)
        if (keys.size < 20) return DEFAULT
        val bottom = keys.maxOf { it.top }
        return keys.filter { k ->
            k.top == bottom && k.top < space.top &&
                // at least half the key is over the space bar
                minOf(k.left + k.w, space.left + space.w) - maxOf(k.left, space.left) >= k.w / 2f
        }.map { it.letter }.toSet().ifEmpty { DEFAULT }
    }
    private val DEFAULT = setOf('c', 'v', 'b', 'n')

    /**
     * The best way to split [typed] into two words at one of [letters], or null. Only when
     * [typed] itself isn't a word and both halves are; single-letter halves only "a" and "I".
     * Prefers the split whose shorter half is longest ("thisnis" -> "this is").
     */
    fun bestSplit(typed: String, letters: Set<Char>, isWord: (String) -> Boolean): String? {
        if (typed.length < 4 || !typed.all { it.isLetter() || it == '\'' }) return null
        if (isWord(typed) || isWord(typed.lowercase())) return null
        var best: String? = null
        var bestScore = -1
        for (i in 1 until typed.length - 1) {
            if (typed[i].lowercaseChar() !in letters) continue
            val left = typed.substring(0, i)
            val right = typed.substring(i + 1)
            if (!okPart(left, isWord) || !okPart(right, isWord)) continue
            val score = minOf(left.length, right.length)
            if (score > bestScore) { bestScore = score; best = "$left $right" }
        }
        return best
    }

    private fun okPart(w: String, isWord: (String) -> Boolean): Boolean {
        if (w.length == 1) return w.equals("a", true) || w.equals("i", true)
        if (w.first() == '\'' || w.last() == '\'' && !w.endsWith("s'")) return false
        return isWord(w) || isWord(w.lowercase())
    }
}
