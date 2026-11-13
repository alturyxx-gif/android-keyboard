package org.futo.inputmethod.latin.tap

import kotlinx.serialization.Serializable
import kotlin.math.abs

/**
 * Letter keys of the keyboard at the time of a tap, in key-detection coordinates (pixels,
 * same frame as the tap positions).
 */
@Serializable
data class KeyBoxes(
    val width: Int, val height: Int,
    val letters: String,
    val x: List<Int>, val y: List<Int>, val w: List<Int>, val h: List<Int>,
    /** The space bar's box (x, y, w, h), or empty if unknown */
    val space: List<Int> = emptyList(),
) {
    /** True if (px, py) is on a bottom-row letter key */
    fun onBottomRow(px: Int, py: Int): Boolean {
        if (y.isEmpty()) return false
        val bottomTop = y.max()
        for (i in letters.indices) if (y[i] == bottomTop && px >= x[i] && px < x[i] + w[i] && py >= y[i] && py < y[i] + h[i]) return true
        return false
    }

    fun index(c: Char): Int = letters.indexOf(c.lowercaseChar())
    fun centre(i: Int): Pair<Float, Float> = (x[i] + w[i] / 2f) to (y[i] + h[i] / 2f)
    fun keyWidth(): Float = if (w.isEmpty()) 1f else w.average().toFloat()
}

/** A tapped word, as stored and exported. */
@Serializable
data class TapRecord(
    val v: Int = 1,
    val time: Long,
    /** "word": a whole tapped word;  "fix": one wrong letter deleted and replaced mid-word. */
    val kind: String,
    /** Letters as the keyboard read your taps. For "fix": the letter that was deleted. */
    val typed: String,
    /** The word that ended up in the text. For "fix": the replacement letter. */
    val word: String,
    /** accepted, autocorrected, picked, reverted (undid autocorrect), fix */
    val source: String,
    /** Tap positions, one per letter of [typed]. */
    val tapX: List<Int>,
    val tapY: List<Int>,
    /** For "fix": where the replacement letter was tapped. */
    val fixX: Int = 0,
    val fixY: Int = 0,
    val context: List<String> = emptyList(),
    val keys: KeyBoxes,
    /** For "fix": the letters of the word typed before the deleted letter */
    val prefix: String = "",
)

object TapSource {
    const val ACCEPTED = "accepted"          // you kept exactly what you typed
    const val AUTOCORRECTED = "autocorrected" // autocorrect changed it and you kept that
    const val PICKED = "picked"              // you tapped a suggestion
    const val REVERTED = "reverted"          // you undid an autocorrection: typed was right
    const val FIX = "fix"                    // a letter deleted and retyped mid-word
}

/** A committed word, before we know whether you keep it. */
data class TapCommit(
    val time: Long,
    val typed: String,
    val committed: String,
    val manualPick: Boolean,
    val tapX: IntArray,
    val tapY: IntArray,
    val context: List<String>,
    val keys: KeyBoxes,
)

/**
 * Works out what each tapped word was meant to be (Tap plan, Stage 0).
 *
 *  - A committed word waits until the next word (or leaving the field). If you undo an
 *    autocorrection with backspace in the meantime, the letters you typed were right ("reverted").
 *    If you backspace into the word to edit it, it's dropped (unclear).
 *  - Mid-word fixes: deleting the last letter and then typing a different letter in the same
 *    place, close to where the deleted tap was, records the deleted tap as a mistake.
 *
 * Not thread-safe; the caller synchronises.
 */
class TapLabeler {
    companion object {
        /** A replacement tap must be within this many key widths of the deleted tap. */
        const val FIX_MAX_KEYS = 1.6f
    }

    private var pending: TapCommit? = null
    /** Backspace went into the committed word: unless it turns out to be an autocorrect undo, drop it */
    private var pendingEdited = false

    private data class Deleted(val letter: Char, val x: Int, val y: Int, val index: Int, val prefix: String)
    private var lastDeleted: Deleted? = null

    fun onCommit(c: TapCommit): List<TapRecord> {
        val out = finalizePending()
        lastDeleted = null
        pending = if (usable(c)) c else null
        pendingEdited = false
        return out
    }

    /** Backspace undid the last autocorrection: what you typed was what you meant. */
    fun onRevert(): List<TapRecord> {
        val p = pending ?: return emptyList()
        pending = null
        pendingEdited = false
        return listOf(record(p, p.typed, TapSource.REVERTED))
    }

    /**
     * Backspace after an already committed word. If it's not an autocorrect undo (which calls
     * [onRevert] right after), you're editing the word: unclear, so it's dropped.
     */
    fun onEditCommitted() {
        pendingEdited = true
    }

    fun onFinishInput(): List<TapRecord> {
        lastDeleted = null
        return finalizePending()
    }

    /** The last letter of the word being typed was deleted (it was at [index]). */
    fun onLetterDeleted(letter: Char, x: Int, y: Int, index: Int, prefix: String = "") {
        lastDeleted = if (letter.isLetter()) Deleted(letter.lowercaseChar(), x, y, index, prefix) else null
    }

    /**
     * Space was typed right after deleting the last letter: if that letter was on the bottom row
     * (just above the space bar), it was a missed space.
     */
    fun onSpaceTyped(index: Int, time: Long, context: List<String>, keys: KeyBoxes?): TapRecord? {
        val d = lastDeleted
        lastDeleted = null
        if (d == null || keys == null || index != d.index || !keys.onBottomRow(d.x, d.y)) return null
        return TapRecord(time = time, kind = "fix", typed = d.letter.toString(), word = " ",
            source = TapSource.FIX, tapX = listOf(d.x), tapY = listOf(d.y), context = context, keys = keys,
            prefix = d.prefix)
    }

    /** A letter was typed at [index] of the word being typed. */
    fun onLetterTyped(letter: Char, x: Int, y: Int, index: Int, time: Long, context: List<String>, keys: KeyBoxes?): TapRecord? {
        val d = lastDeleted
        lastDeleted = null
        if (d == null || keys == null || index != d.index || !letter.isLetter()) return null
        val l = letter.lowercaseChar()
        if (l == d.letter || keys.index(l) < 0 || keys.index(d.letter) < 0) return null
        val kw = keys.keyWidth()
        if (abs(x - d.x) > FIX_MAX_KEYS * kw || abs(y - d.y) > FIX_MAX_KEYS * kw) return null
        return TapRecord(time = time, kind = "fix", typed = d.letter.toString(), word = l.toString(),
            source = TapSource.FIX, tapX = listOf(d.x), tapY = listOf(d.y), fixX = x, fixY = y,
            context = context, keys = keys, prefix = d.prefix)
    }

    private fun finalizePending(): List<TapRecord> {
        val p = pending ?: return emptyList()
        pending = null
        if (pendingEdited) { pendingEdited = false; return emptyList() }
        val source = when {
            p.manualPick -> TapSource.PICKED
            p.committed.equals(p.typed, ignoreCase = true) -> TapSource.ACCEPTED
            else -> TapSource.AUTOCORRECTED
        }
        return listOf(record(p, p.committed, source))
    }

    private fun usable(c: TapCommit): Boolean =
        c.typed.length >= 2 && c.tapX.size == c.typed.length && c.typed.all { it.isLetter() || it == '\'' } &&
                c.tapX.all { it >= 0 } && c.tapY.all { it >= 0 }

    private fun record(p: TapCommit, word: String, source: String) = TapRecord(
        time = p.time, kind = "word", typed = p.typed, word = word, source = source,
        tapX = p.tapX.toList(), tapY = p.tapY.toList(), context = p.context, keys = p.keys,
    )
}
