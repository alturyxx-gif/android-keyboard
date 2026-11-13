package org.futo.inputmethod.latin.tap

import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/** One letter key: box in pixels (key-detection frame). */
data class LetterKey(val letter: Char, val left: Float, val top: Float, val w: Float, val h: Float) {
    val cx get() = left + w / 2f
    val cy get() = top + h / 2f
    fun contains(x: Float, y: Float) = x >= left && x < left + w && y >= top && y < top + h

    companion object {
        fun fromBoxes(b: KeyBoxes): List<LetterKey> = b.letters.indices.map { i ->
            LetterKey(b.letters[i], b.x[i].toFloat(), b.y[i].toFloat(), b.w[i].toFloat(), b.h[i].toFloat())
        }
    }
}

/**
 * Where *you* tend to hit each key (Tap plan, Stage 2), as offsets from the key centre in
 * key widths/heights, plus how spread out your taps are. Learned slowly from words you kept.
 */
@Serializable
class TouchModel(
    val stats: MutableMap<Char, KeyTouch> = HashMap(),
) {
    // forKey() is called for every hit test: cache its arrays instead of allocating each time
    @kotlinx.serialization.Transient private val cache = HashMap<Char, FloatArray>()

    @Serializable
    data class KeyTouch(var n: Int = 0, var mx: Float = 0f, var my: Float = 0f, var vx: Float = 0f, var vy: Float = 0f)

    companion object {
        /** Taps needed on a key before its learned offset is used */
        const val MIN_TAPS = 20
        /** Learning rate floor: older taps fade out slowly (about the last 200 taps per key count) */
        const val MIN_RATE = 0.005f
        /** Taps further than this from the centre (in key sizes) are ignored as outliers */
        const val OUTLIER = 0.9f
        const val DEFAULT_SD_X = 0.30f
        const val DEFAULT_SD_Y = 0.32f
        val NEUTRAL = floatArrayOf(0f, 0f, DEFAULT_SD_X, DEFAULT_SD_Y)
    }

    /** Learn from one tap that was meant for [letter] ([dx], [dy] in key widths/heights). */
    fun learn(letter: Char, dx: Float, dy: Float) {
        if (abs(dx) > OUTLIER || abs(dy) > OUTLIER) return
        val s = stats.getOrPut(letter.lowercaseChar()) { KeyTouch() }
        synchronized(cache) { cache.remove(letter.lowercaseChar()) }
        s.n++
        val r = max(1f / s.n, MIN_RATE)
        val ex = dx - s.mx
        val ey = dy - s.my
        s.mx += r * ex
        s.my += r * ey
        s.vx = (1 - r) * (s.vx + r * ex * ex)
        s.vy = (1 - r) * (s.vy + r * ey * ey)
    }

    /** (offset x, offset y, spread x, spread y) in key sizes; neutral until [MIN_TAPS] taps. */
    fun forKey(letter: Char): FloatArray {
        val l = letter.lowercaseChar()
        synchronized(cache) { cache[l]?.let { return it } }
        val v = compute(l)
        synchronized(cache) { cache[l] = v }
        return v
    }

    private fun compute(letter: Char): FloatArray {
        val s = stats[letter]
        if (s == null || s.n < MIN_TAPS) return NEUTRAL
        // Blend towards the defaults while there are few taps, and keep the spread sensible
        val trust = ((s.n - MIN_TAPS) / 80f).coerceIn(0.3f, 1f)
        return floatArrayOf(
            s.mx * trust, s.my * trust,
            sqrt(s.vx).coerceIn(0.18f, 0.45f) * trust + DEFAULT_SD_X * (1 - trust),
            sqrt(s.vy).coerceIn(0.18f, 0.45f) * trust + DEFAULT_SD_Y * (1 - trust),
        )
    }

    fun tapsLearned(): Int = stats.values.sumOf { it.n }
    fun keysLearned(): Int = stats.values.count { it.n >= MIN_TAPS }
}

/**
 * The key-hit decision (Tap plan, Stages 1 and 4), as pure code so it can be replayed and tested.
 *
 *     score(key) = how well the tap fits the key (your touch model) + strength * ln P(letter next)
 *
 * Safety: a tap in the centre zone of the key under the finger always gives that key, and a tap
 * can only move to a key next to it.
 */
object TapDecision {
    data class Params(
        val strength: Float = 0.5f,
        /** Fraction of the key (width and height) around its centre that is never changed */
        val centreZone: Float = 0.6f,
        /**
         * Added to every probability before taking the log: when no letter is likely (e.g. the
         * word is probably finished) the differences between letters become small.
         */
        val smooth: Float = 0.1f,
        /** A bottom-row tap can only become space when the word is at least this likely finished */
        val minSpaceProb: Float = 0.5f,
    )

    data class Result(val letter: Char, val underFinger: Char, val changed: Boolean)

    /** How far outside a letter key (fraction of a key) a tap can be and still be moved */
    const val MAX_GAP = 0.15f

    /** The key under the finger: the box containing the tap, else the nearest centre. */
    fun underFinger(keys: List<LetterKey>, x: Float, y: Float): LetterKey? =
        keys.firstOrNull { it.contains(x, y) } ?: keys.minByOrNull { dist2(it.cx - x, it.cy - y) }

    /** FUTO's dictionary boost: keys in [boosted] count if the tap is within half a key of their edge. */
    fun futoRule(keys: List<LetterKey>, x: Float, y: Float, boosted: Set<Char>?): Char? {
        if (!boosted.isNullOrEmpty()) {
            var best: LetterKey? = null
            var bestD = Float.MAX_VALUE
            for (k in keys) {
                if (k.letter !in boosted) continue
                val ex = x.coerceIn(k.left, k.left + k.w)
                val ey = y.coerceIn(k.top, k.top + k.h)
                val d = sqrt(dist2(x - ex, y - ey))
                if (d > minOf(k.w, k.h) * 0.5f) continue
                if (d < bestD) { bestD = d; best = k }
            }
            if (best != null) {
                // FUTO then lets a non-boosted key win only if the tap is closer to its centre
                val under = underFinger(keys, x, y)
                if (under != null && under.contains(x, y) && under.letter !in boosted) {
                    val dUnder = sqrt(dist2(under.cx - x, under.cy - y))
                    if (dUnder < bestD) return under.letter
                }
                return best.letter
            }
        }
        return underFinger(keys, x, y)?.letter
    }

    /**
     * Context-aware rule. [probs]: 26 probabilities for a..z (null = no information: the key
     * under the finger, adjusted only by your touch model).
     */
    fun contextRule(
        keys: List<LetterKey>, x: Float, y: Float, probs: FloatArray?, touch: TouchModel?, params: Params,
        space: LetterKey? = null,
    ): Result? {
        val under = underFinger(keys, x, y) ?: return null
        // Only taps on a letter key or in the narrow gaps between letter keys can be moved; in the
        // keyboard's outer margins there's nothing to compare against, so leave them alone
        if (!under.contains(x, y)) {
            val near = keys.minOf { k ->
                val ex = x.coerceIn(k.left, k.left + k.w); val ey = y.coerceIn(k.top, k.top + k.h)
                maxOf(abs(x - ex) / k.w, abs(y - ey) / k.h)
            }
            if (near > MAX_GAP) return Result(under.letter, under.letter, false)
        }
        val uo = touch?.forKey(under.letter) ?: TouchModel.NEUTRAL
        val ucx = under.cx + uo[0] * under.w
        val ucy = under.cy + uo[1] * under.h
        // Centre zone: never changed
        if (abs(x - ucx) < params.centreZone / 2f * under.w && abs(y - ucy) < params.centreZone / 2f * under.h) {
            return Result(under.letter, under.letter, false)
        }
        var best = under
        var bestScore = Float.NEGATIVE_INFINITY
        for (k in keys) {
            if (k !== under && (abs(k.cx - under.cx) > 1.6f * under.w || abs(k.cy - under.cy) > 1.6f * under.h)) continue
            val o = touch?.forKey(k.letter) ?: TouchModel.NEUTRAL
            val dx = (x - (k.cx + o[0] * k.w)) / (o[2] * k.w)
            val dy = (y - (k.cy + o[1] * k.h)) / (o[3] * k.h)
            var score = -0.5f * (dx * dx + dy * dy)
            if (probs != null) {
                val i = k.letter.lowercaseChar() - 'a'
                val p = if (i in 0..25) probs[i] else 0f
                score += params.strength * ln(p + params.smooth)
            }
            if (score > bestScore) { bestScore = score; best = k }
        }
        // Missed space: a tap on the lower part of a bottom-row letter directly above the space
        // bar, when the word is probably finished, can be a space
        if (space != null && probs != null && probs.size > 26 && probs[26] >= params.minSpaceProb &&
            x >= space.left && x < space.left + space.w && y > under.cy &&
            space.top - (under.top + under.h) < 0.5f * under.h && space.top > under.top) {
            val so = touch?.forKey(under.letter) ?: TouchModel.NEUTRAL
            val dy = (y - space.cy) / (so[3] * under.h)
            val spaceScore = -0.5f * dy * dy + params.strength * ln(probs[26] + params.smooth)
            if (spaceScore > bestScore) return Result(' ', under.letter, true)
        }
        return Result(best.letter, under.letter, best.letter != under.letter)
    }

    private fun dist2(dx: Float, dy: Float) = dx * dx + dy * dy
}
