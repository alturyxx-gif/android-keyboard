package org.futo.inputmethod.latin.swipe

import kotlinx.serialization.Serializable
import kotlin.math.abs

/** One finger path, in keyboard-relative coordinates (0..1 across, 0..1 down the letter area). */
@Serializable
data class SwipePath(val x: List<Float>, val y: List<Float>, val t: List<Float>)

/** Letter key centres, in the same coordinates as [SwipePath]. */
@Serializable
data class SwipeKeys(val letters: String, val x: List<Float>, val y: List<Float>) {
    fun centreOf(c: Char): Pair<Float, Float>? {
        val i = letters.indexOf(c.lowercaseChar())
        return if (i < 0) null else x[i] to y[i]
    }
}

/** Exactly what the swipe encoder was given (for replaying the swipe through other models). */
@Serializable
data class SwipeModelFrame(
    val letters: String, val xs: List<Float>, val ys: List<Float>,
    val sx: Float, val sy: Float, val ox: Float, val oy: Float,
    val refiner: Boolean, val contextModel: Boolean,
)

/** A swipe as captured when it finished, before we know which word was meant. */
data class SwipeCapture(
    val timeMillis: Long,
    val path: SwipePath,
    val keys: SwipeKeys,
    val frame: SwipeModelFrame?,
    val candidates: List<String>,
    val context: List<String>,
)

/** A labelled swipe, as stored and exported. */
@Serializable
data class SwipeRecord(
    val v: Int = 1,
    val time: Long,
    /** The word you meant. */
    val word: String,
    /** How [word] was decided: accepted, picked, retyped. */
    val source: String,
    /** What the keyboard offered (first = the word it inserted). */
    val candidates: List<String>,
    /** Up to 10 previous words in the sentence. */
    val context: List<String>,
    val path: SwipePath,
    val keys: SwipeKeys,
    val frame: SwipeModelFrame? = null,
)

object SwipeSource {
    /** You kept the word the keyboard inserted. */
    const val ACCEPTED = "accepted"
    /** You tapped a different suggestion instead. */
    const val PICKED = "picked"
    /** You deleted the swiped word and typed or swiped the right one. */
    const val RETYPED = "retyped"
}

/**
 * Works out which word each swipe was meant to be (Stage 1 of the swipe plan).
 *
 *  - The swipe is followed by a committed word: if it's the inserted word -> accepted; if it's
 *    one of the other candidates -> picked; anything else -> dropped (unclear).
 *  - Backspace deletes the swiped word ("rejected"): the swipe waits for the next committed
 *    word (typed or swiped). That word becomes its label only if it is plausible for the path
 *    (first and last letters near where the finger started and ended), it is committed within
 *    [RETYPE_WINDOW_MS], and it's the very next word. Otherwise the swipe is dropped.
 *  - Swipes in private/no-learning fields are never passed in (checked by the caller).
 *
 * Not thread-safe; the caller synchronises.
 */
class SwipeLabeler(private val clock: () -> Long = System::currentTimeMillis) {
    companion object {
        const val RETYPE_WINDOW_MS = 30_000L
        const val MAX_WAITING = 3
        /** How far (fraction of the keyboard) the path ends may be from the word's first/last key. */
        const val MAX_END_DX = 0.17f
        const val MAX_END_DY = 0.30f
    }

    private var current: SwipeCapture? = null
    private val rejected = ArrayDeque<SwipeCapture>()

    /** A swipe just finished. */
    fun onSwipe(capture: SwipeCapture) {
        // A still-unresolved previous swipe can't be labelled reliably any more
        current = capture
    }

    /** The swiped word was deleted with backspace. */
    fun onRejected() {
        val c = current ?: return
        current = null
        rejected.addLast(c)
        while (rejected.size > MAX_WAITING) rejected.removeFirst()
    }

    /** A word was committed (typed, swiped and accepted, or picked from the strip). */
    fun onCommitted(word: String): List<SwipeRecord> {
        val out = ArrayList<SwipeRecord>()
        val now = clock()
        val w = word.trim()
        if (w.isEmpty()) return out

        // Earlier swipes that were deleted, waiting for the word that replaced them
        for (r in rejected) {
            if (now - r.timeMillis <= RETYPE_WINDOW_MS && isPlausible(r, w)) {
                out.add(record(r, w, SwipeSource.RETYPED))
            }
        }
        rejected.clear()

        current?.let { c ->
            val inserted = c.candidates.firstOrNull()
            when {
                inserted != null && inserted.equals(w, ignoreCase = true) ->
                    out.add(record(c, w, SwipeSource.ACCEPTED))
                c.candidates.any { it.equals(w, ignoreCase = true) } ->
                    out.add(record(c, w, SwipeSource.PICKED))
                else -> { /* unclear: dropped */ }
            }
        }
        current = null
        return out
    }

    /** Leaving the text field: anything unresolved is dropped. */
    fun onFinishInput() {
        current = null
        rejected.clear()
    }

    /** First and last letters of [word] are near where the finger started and ended. */
    fun isPlausible(c: SwipeCapture, word: String): Boolean {
        val letters = word.lowercase().filter { it.isLetter() }
        if (letters.isEmpty() || c.path.x.isEmpty()) return false
        val first = c.keys.centreOf(letters.first()) ?: return false
        val last = c.keys.centreOf(letters.last()) ?: return false
        fun near(p: Pair<Float, Float>, x: Float, y: Float) =
            abs(p.first - x) <= MAX_END_DX && abs(p.second - y) <= MAX_END_DY
        return near(first, c.path.x.first(), c.path.y.first()) &&
                near(last, c.path.x.last(), c.path.y.last())
    }

    private fun record(c: SwipeCapture, word: String, source: String) = SwipeRecord(
        time = c.timeMillis, word = word, source = source,
        candidates = c.candidates, context = c.context,
        path = c.path, keys = c.keys, frame = c.frame,
    )
}
