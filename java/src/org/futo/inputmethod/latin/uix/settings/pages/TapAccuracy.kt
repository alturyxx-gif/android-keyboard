package org.futo.inputmethod.latin.uix.settings.pages

import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.futo.inputmethod.latin.tap.StatsSummary
import org.futo.inputmethod.latin.tap.TapContextAware
import org.futo.inputmethod.latin.tap.TapDebugOverlay
import org.futo.inputmethod.latin.tap.TapKeyFeedback
import org.futo.inputmethod.latin.tap.TapPredictor
import org.futo.inputmethod.latin.tap.TapRecording
import org.futo.inputmethod.latin.tap.TapRecordingEnabled
import org.futo.inputmethod.latin.tap.TapReplay
import org.futo.inputmethod.latin.tap.TapStats
import org.futo.inputmethod.latin.tap.TapStrength
import org.futo.inputmethod.latin.uix.setSetting
import org.futo.inputmethod.latin.uix.settings.ScreenTitle
import org.futo.inputmethod.latin.uix.settings.ScrollableList
import org.futo.inputmethod.latin.uix.settings.SettingSlider
import org.futo.inputmethod.latin.uix.settings.SettingToggleDataStore
import org.futo.inputmethod.latin.uix.settings.Tip
import org.futo.inputmethod.latin.uix.settings.useDataStoreValue
import kotlin.math.roundToInt

private fun statsText(s: StatsSummary?): String = if (s == null) "No typing recorded in this period." else buildString {
    append("${s.letters} letters, ${s.words} words over ${s.days} day(s)\n")
    append("Backspaces: ${"%.1f".format(s.backspacesPer100Letters)} per 100 letters\n")
    append("Mid-word fixes (deleted and retyped a letter): ${"%.2f".format(s.fixesPer100Letters)} per 100 letters\n")
    append("Words autocorrected: ${"%.1f".format(s.autocorrectedPct)}% · undone: ${"%.1f".format(s.revertedPct)}%")
    if (s.changedTaps > 0) {
        append("\nTaps moved to a likelier key: ${s.changedTaps} (deleted straight away: ${"%.0f".format(s.changedTapsUndonePct)}%)")
    }
}

@Composable
fun TapAccuracyScreen(navController: NavHostController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var refresh by remember { mutableIntStateOf(0) }

    val contextAware = useDataStoreValue(TapContextAware)
    val strength = useDataStoreValue(TapStrength)
    val debug = useDataStoreValue(TapDebugOverlay)
    LaunchedEffect(contextAware, strength, debug) { TapPredictor.refreshSettings(context) }

    val week = remember(refresh) { TapStats.summary(context, 7) }
    val prevWeek = remember(refresh) { TapStats.summary(context, 7, skipLast = 7) }
    val touch = remember(refresh) { TapPredictor.touchModel(context) }

    LaunchedEffect(Unit) { withContext(Dispatchers.IO) { TapRecording.ensureCountLoaded(context) } }
    val count by TapRecording.count.collectAsState()

    var running by remember { mutableStateOf(false) }
    var progress by remember { mutableStateOf(0f) }
    var report by remember { mutableStateOf<TapReplay.Report?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }

    ScrollableList {
        ScreenTitle("Tap accuracy", showBack = true, navController)

        ScreenTitle("Last 7 days")
        Tip(statsText(week))
        if (prevWeek != null) {
            ScreenTitle("The 7 days before")
            Tip(statsText(prevWeek))
        }

        ScreenTitle("Key detection")
        Tip("Taps near the edge between two keys go to the letter you most likely meant, using " +
                "where you usually tap, the word so far and the sentence so far. Taps in the middle " +
                "of a key are never changed. Needs \"Smart key-hit detection\" (Text Prediction) on.")
        SettingToggleDataStore(
            title = "Context-aware key detection",
            subtitle = "Off = FUTO's dictionary boost only",
            setting = TapContextAware
        )
        if (contextAware) {
            SettingSlider(
                title = "Strength",
                setting = TapStrength,
                range = 0.0f..1.5f,
                transform = { (it * 20).roundToInt() / 20f },
                indicator = { "${"%.2f".format(it)}" },
                subtitle = "How strongly likely letters pull taps near key edges. Lowered automatically " +
                        "while you keep deleting the letters it picks (now ×${"%.2f".format(TapKeyFeedback.strengthMultiplier())})."
            )
        }
        SettingToggleDataStore(
            title = "Show key areas (testing)",
            subtitle = "Shades where taps now go to a different key, and dots where you usually hit each key",
            setting = TapDebugOverlay
        )
        Tip("Your touch model: ${touch.tapsLearned()} taps learned, ${touch.keysLearned()} keys adjusted " +
                "(a key needs 20 taps before its offset is used).")
        OutlinedButton(modifier = Modifier.padding(horizontal = 16.dp), onClick = {
            TapPredictor.resetTouchModel(context)
            TapKeyFeedback.reset()
            refresh++
        }) { Text("Reset touch model") }

        ScreenTitle("Compare on my taps")
        Tip("Replays your recorded taps through each rule and counts wrong letters per 100 taps, " +
                "then finds the best strength for you.")
        Button(modifier = Modifier.padding(horizontal = 16.dp), enabled = !running, onClick = {
            running = true; progress = 0f; report = null; error = null
            scope.launch {
                try {
                    report = withContext(Dispatchers.Default) { TapReplay.run(context) { progress = it } }
                } catch (e: IllegalStateException) {
                    error = e.message
                } catch (e: Throwable) {
                    error = "Comparison failed: ${e.message}"
                }
                running = false
            }
        }) { Text("Compare") }
        if (running) LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().padding(16.dp))
        error?.let { Tip(it) }
        report?.let { r ->
            Tip(r.rows.joinToString("\n") { "${it.name}: ${"%.2f".format(it.wrongPer100)} wrong per 100" } + "\n\n" + r.note)
            r.bestStrength?.let { best ->
                Button(modifier = Modifier.padding(horizontal = 16.dp), onClick = {
                    scope.launch {
                        context.setSetting(TapStrength, best)
                        TapPredictor.refreshSettings(context)
                        Toast.makeText(context, "Strength set to ${"%.2f".format(best)}", Toast.LENGTH_SHORT).show()
                    }
                }) { Text("Use best strength (${"%.2f".format(best)})") }
            }
        }

        ScreenTitle("Record my taps")
        Tip("When on, each tapped word is saved with where you tapped and the word you kept, for the " +
                "comparison above and tuning. Stored encrypted on the phone; never in password or " +
                "incognito fields; nothing is uploaded. The statistics above are always kept (numbers only).")
        SettingToggleDataStore(
            title = "Record my taps for tuning",
            subtitle = "$count recorded",
            setting = TapRecordingEnabled
        )
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(enabled = count > 0, onClick = {
                scope.launch {
                    val where = withContext(Dispatchers.IO) { TapRecording.export(context) }
                    Toast.makeText(context, if (where != null) "Saved to $where (not encrypted: delete after copying)" else "Export failed", Toast.LENGTH_LONG).show()
                }
            }) { Text("Export") }
            OutlinedButton(enabled = count > 0, onClick = { confirmDelete = true }) { Text("Delete all") }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete recorded taps?") },
            text = { Text("This deletes all $count recorded taps. Statistics and your touch model are kept.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    scope.launch { withContext(Dispatchers.IO) { TapRecording.deleteAll(context) } }
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } }
        )
    }
}
