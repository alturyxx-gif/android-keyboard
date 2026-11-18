package org.futo.inputmethod.latin.xlm

import android.os.SystemClock
import android.view.Choreographer
import androidx.datastore.preferences.core.booleanPreferencesKey
import org.futo.inputmethod.latin.uix.SettingsKey

/** Show model load timings and UI stalls on the keyboard itself (for testing). */
val ShowModelTimings = SettingsKey(booleanPreferencesKey("show_model_timings"), false)

/**
 * When and why the typing model (and the key predictor's copy) were loaded, and how long it
 * took, so hitches can be matched to loads.
 */
object ModelLoadStats {
    data class Event(val which: String, val reason: String, val ms: Long, val ok: Boolean, val atMs: Long)

    @Volatile var enabled = false
    @Volatile var version = 0
        private set
    @Volatile private var events: List<Event> = emptyList()
    @Volatile private var loadingNow: Pair<String, Long>? = null

    fun started(which: String, reason: String) {
        loadingNow = "$which ($reason)" to SystemClock.elapsedRealtime()
        version++
    }

    fun finished(which: String, reason: String, ms: Long, ok: Boolean) {
        loadingNow = null
        events = (events + Event(which, reason, ms, ok, SystemClock.elapsedRealtime())).takeLast(6)
        version++
    }

    /** Most recent load per model, e.g. "typing model 820 ms (keyboard opened, 2 min ago)". */
    fun summary(): String {
        val now = SystemClock.elapsedRealtime()
        val parts = ArrayList<String>()
        loadingNow?.let { (what, t) -> parts.add("loading $what… ${now - t} ms") }
        for (e in events.reversed().distinctBy { it.which }.reversed()) {
            val ago = (now - e.atMs) / 1000
            val agoText = if (ago < 60) "${ago}s ago" else "${ago / 60} min ago"
            parts.add("${e.which} ${if (e.ok) "${e.ms} ms" else "FAILED"} (${e.reason}, $agoText)")
        }
        return if (parts.isEmpty()) "no model loaded yet in this session" else parts.joinToString(" · ")
    }
}

/**
 * Measures how long the keyboard's UI thread goes without drawing a frame (a hitch). Runs only
 * while timings are shown and the keyboard is visible.
 */
object UiStallMonitor {
    private const val WINDOW_MS = 10_000L
    private const val STALL_MS = 50L
    @Volatile var running = false
        private set
    private var last = 0L
    private val stalls = ArrayDeque<Pair<Long, Long>>() // (time, length ms)
    @Volatile var version = 0
        private set
    private var isVisible: () -> Boolean = { false }

    private val callback = object : Choreographer.FrameCallback {
        override fun doFrame(frameTimeNanos: Long) {
            val t = frameTimeNanos / 1_000_000
            if (last != 0L && t - last > STALL_MS) {
                synchronized(stalls) { stalls.addLast(t to (t - last)) }
                version++
            }
            last = t
            if (ModelLoadStats.enabled && isVisible()) Choreographer.getInstance().postFrameCallback(this)
            else { running = false; last = 0L }
        }
    }

    /** Call on the main thread (e.g. from the keyboard view's onDraw). */
    fun ensureRunning(visible: () -> Boolean) {
        isVisible = visible
        if (running || !ModelLoadStats.enabled) return
        running = true
        last = 0L
        Choreographer.getInstance().postFrameCallback(callback)
    }

    fun summary(): String {
        val now = SystemClock.uptimeMillis()
        val recent = synchronized(stalls) {
            while (stalls.isNotEmpty() && now - stalls.first().first > WINDOW_MS) stalls.removeFirst()
            stalls.toList()
        }
        if (recent.isEmpty()) return "UI: no stalls in the last 10 s"
        return "UI: longest stall ${recent.maxOf { it.second }} ms, ${recent.size} over $STALL_MS ms in the last 10 s"
    }
}
