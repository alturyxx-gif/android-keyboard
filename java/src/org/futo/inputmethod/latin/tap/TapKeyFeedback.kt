package org.futo.inputmethod.latin.tap

import android.content.Context

/**
 * Notices when you immediately delete a letter that the context-aware key detection chose
 * instead of the key under your finger (Tap plan, Stage 4.3). Counted in [TapStats], and used
 * to back off the strength automatically if it keeps happening.
 */
object TapKeyFeedback {
    private enum class State { NONE, CHANGED_TAP, AWAITING_NEXT }
    @Volatile private var state = State.NONE

    /** Recent outcomes of changed taps: true = you deleted it straight away. */
    private val recent = ArrayDeque<Boolean>()
    private const val WINDOW = 50

    /** Called by key detection when it picked a different key than the one under the finger. */
    @JvmStatic
    fun onTapChanged() { state = State.CHANGED_TAP }

    fun onLetterTyped(context: Context) {
        state = when (state) {
            State.CHANGED_TAP -> { TapStats.onChangedTap(context); State.AWAITING_NEXT }
            State.AWAITING_NEXT -> { record(false); State.NONE }
            State.NONE -> State.NONE
        }
    }

    /** A space was typed (it may itself be a changed tap: a bottom-row letter moved to space). */
    fun onSpaceTyped(context: Context) = onLetterTyped(context)

    fun onBackspace(context: Context) {
        if (state == State.AWAITING_NEXT || state == State.CHANGED_TAP) {
            TapStats.onChangedTapUndone(context)
            record(true)
        }
        state = State.NONE
    }

    private fun record(undone: Boolean) = synchronized(recent) {
        recent.addLast(undone)
        while (recent.size > WINDOW) recent.removeFirst()
    }

    /**
     * Multiplier (0.4..1) on the configured strength: lowered while many recent changed taps
     * were deleted straight away, recovers as they stop being deleted.
     */
    fun strengthMultiplier(): Float = synchronized(recent) {
        if (recent.size < 10) return 1f
        val undoneRate = recent.count { it }.toFloat() / recent.size
        // Up to ~15% undone is normal (people also delete correct letters); above that, back off
        return (1f - (undoneRate - 0.15f).coerceAtLeast(0f) * 2f).coerceIn(0.4f, 1f)
    }

    fun reset() = synchronized(recent) { recent.clear(); state = State.NONE }
}
