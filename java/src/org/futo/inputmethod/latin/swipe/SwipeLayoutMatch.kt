package org.futo.inputmethod.latin.swipe

import kotlin.math.abs

/**
 * Result of comparing the keyboard's letter positions with the layout a swipe "refiner"
 * (FUTO's layout-specific decoder) was trained on.
 */
data class LayoutMatchResult(
    val matches: Boolean,
    /** Plain-language explanation, shown in the swipe status box. */
    val reason: String,
    /** Largest difference found, as a fraction of the keyboard's letter area. */
    val maxDx: Float = 0f,
    val maxDy: Float = 0f,
    val worstKey: Char? = null,
    val offsetX: Float = 0f,
    val offsetY: Float = 0f,
    val scaleX: Float = 0f,
    val scaleY: Float = 0f,
)

/**
 * Pure maths behind the layout-refiner matching (kept free of Android types so it can be unit
 * tested). Key centres are normalised so the left-most/top-most letter is 0 and the
 * right-most/bottom-most is 1, which makes the comparison independent of keyboard height,
 * width and padding.
 */
object SwipeLayoutMatch {
    fun compare(
        letters: String,
        referenceXs: List<Float>,
        referenceYs: List<Float>,
        maxDeviationX: Float,
        maxDeviationY: Float,
        positions: List<Pair<Float, Float>>,
    ): LayoutMatchResult {
        require(letters.length == referenceXs.size && letters.length == referenceYs.size)
        if (positions.size != letters.length) {
            return LayoutMatchResult(false, "Keyboard has ${positions.size} of the ${letters.length} letters")
        }
        val minX = positions.minOf { it.first }
        val minY = positions.minOf { it.second }
        val scaleX = positions.maxOf { it.first } - minX
        val scaleY = positions.maxOf { it.second } - minY
        if (scaleX <= 0f || scaleY <= 0f) return LayoutMatchResult(false, "Letter keys have no width or height")

        var maxDx = 0f
        var maxDy = 0f
        var worst: Char? = null
        var worstAmount = -1f
        for (i in letters.indices) {
            val dx = abs((positions[i].first - minX) / scaleX - referenceXs[i])
            val dy = abs((positions[i].second - minY) / scaleY - referenceYs[i])
            if (dx > maxDx) maxDx = dx
            if (dy > maxDy) maxDy = dy
            // Relative to each axis' limit, so the "worst" key is the one closest to failing
            val rel = maxOf(dx / maxDeviationX, dy / maxDeviationY)
            if (rel > worstAmount) { worstAmount = rel; worst = letters[i] }
        }
        val ok = maxDx < maxDeviationX && maxDy < maxDeviationY
        val reason = if (ok) {
            "Key positions match (largest difference ${pct(maxDx)} across, ${pct(maxDy)} down; " +
                    "limits ${pct(maxDeviationX)} / ${pct(maxDeviationY)})"
        } else {
            "Key '${worst}' is too far from where the refiner expects it " +
                    "(${pct(maxDx)} across, ${pct(maxDy)} down; limits ${pct(maxDeviationX)} / ${pct(maxDeviationY)})"
        }
        return LayoutMatchResult(ok, reason, maxDx, maxDy, worst, minX, minY, scaleX, scaleY)
    }

    private fun pct(v: Float) = "${(v * 1000).toInt() / 10f}%"
}
