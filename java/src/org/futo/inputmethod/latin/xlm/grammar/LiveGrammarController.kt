package org.futo.inputmethod.latin.xlm.grammar

import android.content.Context
import android.os.Build
import android.text.SpannableString
import android.text.Spanned
import android.text.style.SuggestionSpan
import android.util.Log
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.lifecycle.LifecycleCoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.futo.inputmethod.latin.BuildConfig
import org.futo.inputmethod.latin.inputlogic.InputLogic
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.inputmethod.latin.xlm.LanguageModel

val LiveGrammarEnabled = SettingsKey(
    booleanPreferencesKey("live_grammar_correction_enabled"),
    true
)

val LiveGrammarAutoApply = SettingsKey(
    booleanPreferencesKey("live_grammar_correction_auto_apply"),
    true
)

/**
 * Runs [GrammarChecker] while you type and applies its fixes to the text field.
 *
 *  - Runs after a word is finished (space or punctuation), ~300 ms after the last keypress.
 *    Typing again cancels the pending check.
 *  - High-confidence fixes are applied automatically. Backspace right afterwards undoes the
 *    fix (and it won't be proposed again for that sentence).
 *  - Low-confidence fixes are underlined using Android's grammar-error SuggestionSpan, so
 *    apps that support it (most standard text fields) show the fix when you tap the word.
 *
 * All edits are made as "delete the tail of the sentence and re-type it", so the cursor
 * never moves and the keyboard's own text cache stays consistent.
 */
class LiveGrammarController(
    private val context: Context,
    private val inputLogic: InputLogic,
    private val lifecycleScope: LifecycleCoroutineScope,
    private val modelProvider: () -> LanguageModel?,
    private val isEnglish: () -> Boolean,
) {
    private val TAG = "LiveGrammar"

    private val checker = GrammarChecker(ContinuationScorer { prefix, continuations ->
        modelProvider()?.scoreContinuations(prefix, continuations)
    })

    private var pending: Job? = null
    private val rejected = LinkedHashSet<String>()

    /** What the last auto-applied fix did, so backspace can undo it. */
    private data class AppliedFix(
        val textAfter: String,   // text before cursor right after the fix
        val tailBefore: String,  // tail as it was before the fix
        val tailAfter: String,   // tail as written by the fix
        val ignoreKey: String,
    )
    private var lastApplied: AppliedFix? = null

    private val enabled get() = context.getSetting(LiveGrammarEnabled)

    fun onStartInput() {
        pending?.cancel()
        lastApplied = null
        rejected.clear()
    }

    fun onFinishInput() {
        pending?.cancel()
        lastApplied = null
    }

    /**
     * Called before an input event is processed. Returns true if the event was consumed
     * (backspace that undid a grammar fix).
     */
    fun onBeforeEvent(keyCode: Int, isDelete: Boolean): Boolean {
        pending?.cancel()
        val applied = lastApplied ?: return false
        lastApplied = null
        if (!isDelete) return false

        val conn = inputLogic.mConnection
        if (inputLogic.mWordComposer.isComposingWord || conn.hasSelection()) return false
        val before = conn.getTextBeforeCursor(applied.textAfter.length + 1, 0)?.toString() ?: return false
        if (!before.endsWith(applied.textAfter)) return false

        conn.beginBatchEdit()
        conn.deleteTextBeforeCursor(applied.tailAfter.length)
        conn.commitText(applied.tailBefore, 1)
        conn.endBatchEdit()
        conn.send()
        rejected.add(applied.ignoreKey)
        if (BuildConfig.DEBUG) Log.d(TAG, "Undid grammar fix via backspace")
        return true
    }

    /** Called after every input event; schedules a check when a word was just finished. */
    fun onAfterEvent() {
        if (!enabled) return
        if (!isEnglish()) return
        val conn = inputLogic.mConnection
        if (inputLogic.mWordComposer.isComposingWord || conn.hasSelection()) return
        val last = conn.getCodePointBeforeCursor()
        if (last == -1) return
        val ch = last.toChar()
        if (!(ch == ' ' || ch == ',' || ch == '.' || ch == '!' || ch == '?' || ch == ';' || ch == ':')) return

        pending?.cancel()
        pending = lifecycleScope.launch {
            delay(300)
            runCheck()
        }
    }

    private suspend fun runCheck() {
        val conn = inputLogic.mConnection
        val snapshot = withContext(Dispatchers.Main) {
            if (inputLogic.mWordComposer.isComposingWord || conn.hasSelection()) null
            else conn.getTextBeforeCursor(400, 0)?.toString()
        } ?: return
        if (snapshot.isBlank()) return

        val fixes = try {
            checker.check(snapshot, rejected)
        } catch (e: Exception) {
            if (e is kotlinx.coroutines.CancellationException) throw e
            Log.e(TAG, "Grammar check failed", e)
            return
        }
        if (fixes.isEmpty()) return

        withContext(Dispatchers.Main) { applyFixes(snapshot, fixes) }
    }

    private fun applyFixes(snapshot: String, fixes: List<GrammarFix>) {
        val conn = inputLogic.mConnection
        if (inputLogic.mWordComposer.isComposingWord || conn.hasSelection()) return
        val now = conn.getTextBeforeCursor(snapshot.length, 0)?.toString() ?: return
        if (now != snapshot) return // user kept typing; a new check will run

        val autoApply = context.getSetting(LiveGrammarAutoApply)
        val firstStart = fixes.minOf { it.start }
        val tailBefore = snapshot.substring(firstStart)

        val newTail = SpannableString(buildTail(snapshot, firstStart, fixes, autoApply))
        var anyChange = false
        var autoKey: String? = null
        // Second pass: positions in the new tail, and spans for underlined fixes
        var shift = 0
        for (fix in fixes.sortedBy { it.start }) {
            val apply = autoApply && fix.confidence == FixConfidence.High
            val startInTail = fix.start - firstStart + shift
            if (apply) {
                shift += fix.replacement.length - fix.original.length
                anyChange = true
                autoKey = checker.ignoreKey(fix.original, fix.replacement, snapshot.substring(0, fix.start))
            } else {
                val endInTail = startInTail + fix.original.length
                newTail.setSpan(grammarSpan(fix.replacement), startInTail, endInTail, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                anyChange = true
            }
        }
        if (!anyChange) return

        conn.beginBatchEdit()
        conn.deleteTextBeforeCursor(tailBefore.length)
        conn.commitText(newTail, 1)
        conn.endBatchEdit()
        conn.send()

        val tailAfter = newTail.toString()
        lastApplied = autoKey?.let {
            AppliedFix(
                textAfter = snapshot.substring(0, firstStart) + tailAfter,
                tailBefore = tailBefore,
                tailAfter = tailAfter,
                ignoreKey = it
            )
        }
        if (BuildConfig.DEBUG) Log.d(TAG, "Applied grammar fixes: $fixes")
    }

    private fun buildTail(snapshot: String, firstStart: Int, fixes: List<GrammarFix>, autoApply: Boolean): String {
        val sb = StringBuilder()
        var pos = firstStart
        for (fix in fixes.sortedBy { it.start }) {
            sb.append(snapshot, pos, fix.start)
            val apply = autoApply && fix.confidence == FixConfidence.High
            sb.append(if (apply) fix.replacement else fix.original)
            pos = fix.end
        }
        sb.append(snapshot, pos, snapshot.length)
        return sb.toString()
    }

    private fun grammarSpan(replacement: String): SuggestionSpan {
        val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            SuggestionSpan.FLAG_GRAMMAR_ERROR or SuggestionSpan.FLAG_EASY_CORRECT
        } else {
            SuggestionSpan.FLAG_MISSPELLED or SuggestionSpan.FLAG_EASY_CORRECT
        }
        return SuggestionSpan(context, arrayOf(replacement), flags)
    }
}
