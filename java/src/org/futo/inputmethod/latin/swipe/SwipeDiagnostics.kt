package org.futo.inputmethod.latin.swipe

import androidx.compose.runtime.mutableStateOf

/**
 * What the swipe engine is doing, for the "Swipe engine" status box in the swipe settings.
 * Updated by the keyboard as you swipe (same process as the settings screen).
 */
object SwipeDiagnostics {
    data class LastSwipe(
        val timeMillis: Long,
        val encoder: String,
        val refinerActive: Boolean,
        val contextModelActive: Boolean,
        val totalMs: Float,
        val encoderMs: Float,
        val refinerMs: Float,
        val searchMs: Float,
        val contextMs: Float,
        val beamWidth: Int,
        val candidates: List<String>,
    )

    /** Why the layout refiner is (or isn't) used for the current keyboard. */
    val refinerMatch = mutableStateOf("Not checked yet — open the keyboard and swipe once")

    /** Details of the most recent completed swipe. */
    val lastSwipe = mutableStateOf<LastSwipe?>(null)
}
