package org.futo.inputmethod.latin.tap

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.time.LocalDate

/** Counts for one day. Only numbers are kept, never what was typed. */
@Serializable
data class DayStats(
    val day: String,
    var letters: Long = 0,
    var backspaces: Long = 0,
    var words: Long = 0,
    var autocorrected: Long = 0,
    var reverted: Long = 0,
    var midWordFixes: Long = 0,
    /** Taps where the new key detection picked a different key than the one under the finger */
    var changedTaps: Long = 0,
    /** ...and then the letter was deleted straight away */
    var changedTapsUndone: Long = 0,
)

data class StatsSummary(
    val days: Int, val letters: Long, val words: Long,
    val backspacesPer100Letters: Float, val autocorrectedPct: Float, val revertedPct: Float,
    val fixesPer100Letters: Float, val changedTaps: Long, val changedTapsUndonePct: Float,
)

/**
 * Tap statistics (Tap plan, Stage 0.2): always on, numbers only, last 60 days, stored in the
 * app's private storage. Shown on the Tap accuracy page to see whether changes help.
 */
object TapStats {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val days = LinkedHashMap<String, DayStats>()
    private var loaded = false
    private var dirty = 0

    private fun file(context: Context) = File(context.filesDir, "tap-stats.json")

    private fun today(context: Context): DayStats = synchronized(this) {
        if (!loaded) {
            try {
                file(context).takeIf { it.isFile }?.readText()?.let { t ->
                    json.decodeFromString<List<DayStats>>(t).forEach { days[it.day] = it }
                }
            } catch (_: Exception) { }
            loaded = true
        }
        val d = LocalDate.now().toString()
        days.getOrPut(d) { DayStats(d) }
    }

    private fun bump(context: Context, f: (DayStats) -> Unit) {
        synchronized(this) {
            f(today(context))
            if (++dirty >= 50) save(context)
        }
    }

    fun onLetter(context: Context) = bump(context) { it.letters++ }
    fun onBackspace(context: Context) = bump(context) { it.backspaces++ }
    fun onWord(context: Context, autocorrected: Boolean) = bump(context) { it.words++; if (autocorrected) it.autocorrected++ }
    fun onReverted(context: Context) = bump(context) { it.reverted++ }
    fun onMidWordFix(context: Context) = bump(context) { it.midWordFixes++ }
    fun onChangedTap(context: Context) = bump(context) { it.changedTaps++ }
    fun onChangedTapUndone(context: Context) = bump(context) { it.changedTapsUndone++ }

    fun save(context: Context) = synchronized(this) {
        dirty = 0
        val keep = days.values.sortedBy { it.day }.takeLast(60)
        days.clear(); keep.forEach { days[it.day] = it }
        try { file(context).writeText(json.encodeToString(keep)) } catch (_: Exception) { }
    }

    /** Totals over the last [n] days (including today), or null if nothing was typed. */
    fun summary(context: Context, n: Int = 7, skipLast: Int = 0): StatsSummary? = synchronized(this) {
        today(context)
        val sel = days.values.sortedBy { it.day }.dropLast(skipLast).takeLast(n)
        val letters = sel.sumOf { it.letters }
        val words = sel.sumOf { it.words }
        if (letters == 0L) return null
        val changed = sel.sumOf { it.changedTaps }
        StatsSummary(
            days = sel.size, letters = letters, words = words,
            backspacesPer100Letters = sel.sumOf { it.backspaces } * 100f / letters,
            autocorrectedPct = if (words == 0L) 0f else sel.sumOf { it.autocorrected } * 100f / words,
            revertedPct = if (words == 0L) 0f else sel.sumOf { it.reverted } * 100f / words,
            fixesPer100Letters = sel.sumOf { it.midWordFixes } * 100f / letters,
            changedTaps = changed,
            changedTapsUndonePct = if (changed == 0L) 0f else sel.sumOf { it.changedTapsUndone } * 100f / changed,
        )
    }

    fun reset(context: Context) = synchronized(this) {
        days.clear(); loaded = true
        file(context).delete()
    }
}
