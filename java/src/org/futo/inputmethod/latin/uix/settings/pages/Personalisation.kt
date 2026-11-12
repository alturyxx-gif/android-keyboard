package org.futo.inputmethod.latin.uix.settings.pages

import android.text.format.DateFormat
import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
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
import org.futo.inputmethod.latin.uix.USE_TRANSFORMER_FINETUNING
import org.futo.inputmethod.latin.uix.settings.ScreenTitle
import org.futo.inputmethod.latin.uix.settings.ScrollableList
import org.futo.inputmethod.latin.uix.settings.SettingItem
import org.futo.inputmethod.latin.uix.settings.SettingSlider
import org.futo.inputmethod.latin.uix.settings.SettingToggleDataStore
import org.futo.inputmethod.latin.uix.settings.Tip
import org.futo.inputmethod.latin.uix.settings.useDataStoreValue
import org.futo.inputmethod.latin.uix.theme.Typography
import org.futo.inputmethod.latin.xlm.ModelPaths
import org.futo.inputmethod.latin.xlm.TrainingWorkerStatus
import org.futo.inputmethod.latin.xlm.personal.PersonalAdapterFiles
import org.futo.inputmethod.latin.xlm.personal.PersonalisationStore
import org.futo.inputmethod.latin.xlm.personal.PersonalisationStrength
import org.futo.inputmethod.latin.xlm.personal.TrainingRunRecord
import org.futo.inputmethod.latin.xlm.personal.TypingModelThreads
import org.futo.inputmethod.latin.xlm.scheduleTrainingWorkerImmediately
import java.util.Date
import kotlin.math.roundToInt

/**
 * "What it learned" screen (Phase 2, Stage 2.1).
 *
 * Shows what personalisation has learned and lets you control it: training history with
 * before/after quality, recently learned words, the Roman Urdu vocabulary, strength,
 * "train now", "undo last training" and "reset".
 */
@Composable
fun PersonalisationScreen(navController: NavHostController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var refresh by remember { mutableIntStateOf(0) }
    val historySize by PersonalisationStore.historySize.collectAsState()
    val isTraining = TrainingWorkerStatus.isTraining.value
    val progress by TrainingWorkerStatus.progress.collectAsState(initial = 0f)

    val enabled = useDataStoreValue(USE_TRANSFORMER_FINETUNING)
    val strength = useDataStoreValue(PersonalisationStrength)
    val threads = useDataStoreValue(TypingModelThreads)

    // Re-load the typing model whenever strength or threads change so it takes effect at once
    LaunchedEffect(strength, threads, enabled) { ModelPaths.signalReloadModels() }

    val data = remember(refresh, historySize) {
        PersonalisationStore.ensureLoaded(context)
        val history = PersonalisationStore.snapshot(context)
        val runs = PersonalisationStore.runs(context).reversed()
        val recentEnglish = history.asReversed().filter { !it.romanUrdu }
            .map { it.committedWord }.distinct().take(60)
        val urdu = PersonalisationStore.urduVocab.learnedWords().entries
            .sortedByDescending { it.value }.take(60)
        Triple(runs, recentEnglish, urdu)
    }
    val (runs, recentWords, urduWords) = data

    var hasPrevious by remember { mutableStateOf(false) }
    LaunchedEffect(refresh) {
        hasPrevious = withContext(Dispatchers.IO) {
            ModelPaths.getModelOptions(context).values.any { PersonalAdapterFiles.hasPrevious(it.path) }
        }
    }

    var confirmReset by remember { mutableStateOf(false) }

    ScrollableList {
        ScreenTitle("Personalisation", showBack = true, navController)

        Tip("Everything here stays on your phone. Your typing history is stored encrypted and " +
                "only the last 60 days are kept. Training runs while the phone is charging.")

        SettingToggleDataStore(
            title = "Learn from my typing",
            subtitle = "Trains a small personal adjustment for the typing model, and remembers new words straight away",
            setting = USE_TRANSFORMER_FINETUNING
        )

        if (enabled) {
            SettingItem(
                title = "Typing history",
                subtitle = "$historySize words remembered · ${urduWords.size} Roman Urdu words"
            ) {}

            SettingSlider(
                title = "Personalisation strength",
                setting = PersonalisationStrength,
                range = 0.0f..1.0f,
                transform = { (it * 20).roundToInt() / 20f },
                indicator = { if (it <= 0f) "Off" else "${(it * 100).roundToInt()}%" },
                subtitle = "How much the learned adjustment changes predictions"
            )

            SettingSlider(
                title = "Typing model threads",
                setting = TypingModelThreads,
                range = 1.0f..4.0f,
                transform = { it.roundToInt() },
                indicator = { "$it" },
                subtitle = "More threads = faster predictions on a fast phone. FUTO's default was 1.",
                steps = 2
            )

            if (isTraining) {
                Column(Modifier.padding(16.dp)) {
                    Text("Training in progress… ${(progress * 100).roundToInt()}%", style = Typography.Body.MediumMl)
                    LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                }
            }

            Row(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(enabled = !isTraining, onClick = {
                    PersonalisationStore.flush(context)
                    scheduleTrainingWorkerImmediately(context)
                    Toast.makeText(context, "Training started", Toast.LENGTH_SHORT).show()
                }) { Text("Train now") }

                OutlinedButton(enabled = !isTraining && hasPrevious, onClick = {
                    scope.launch {
                        withContext(Dispatchers.IO) {
                            ModelPaths.getModelOptions(context).values.forEach { PersonalAdapterFiles.rollBack(it.path) }
                        }
                        ModelPaths.signalReloadModels()
                        refresh++
                        Toast.makeText(context, "Went back to the previous training", Toast.LENGTH_SHORT).show()
                    }
                }) { Text("Undo last training") }
            }

            OutlinedButton(
                modifier = Modifier.padding(horizontal = 16.dp),
                enabled = !isTraining,
                onClick = { confirmReset = true }
            ) { Text("Reset all personalisation") }

            ScreenTitle("Training runs")
            if (runs.isEmpty()) {
                Tip("No training yet. It needs at least ${org.futo.inputmethod.latin.xlm.personal.TrainingDataBuilder.MIN_ENTRIES} words of typing, then runs while charging (or tap Train now).")
            }
            runs.take(15).forEach { run -> TrainingRunItem(run) }

            ScreenTitle("Recently learned words")
            if (recentWords.isEmpty()) Tip("Nothing learned yet.")
            recentWords.forEach { word ->
                SettingItem(title = word, compact = true) {
                    IconButton(onClick = {
                        PersonalisationStore.forgetWord(context, word)
                        refresh++
                    }) { Icon(Icons.Default.Clear, contentDescription = "Forget $word") }
                }
            }

            ScreenTitle("Roman Urdu vocabulary")
            Tip("Words you type in Roman Urdu sentences. Autocorrect leaves these alone, and they're only suggested in Roman Urdu sentences.")
            urduWords.forEach { (word, count) ->
                SettingItem(title = word, subtitle = "used $count×", compact = true) {
                    IconButton(onClick = {
                        PersonalisationStore.forgetWord(context, word)
                        refresh++
                    }) { Icon(Icons.Default.Clear, contentDescription = "Forget $word") }
                }
            }
        }
    }

    if (confirmReset) {
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Reset personalisation?") },
            text = { Text("This deletes your typing history, the Roman Urdu vocabulary and all trained adjustments. The base typing model is not affected.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmReset = false
                    scope.launch {
                        withContext(Dispatchers.IO) {
                            PersonalisationStore.clearAll(context)
                            ModelPaths.getModelOptions(context).values.forEach { PersonalAdapterFiles.deleteAll(it.path) }
                        }
                        ModelPaths.signalReloadModels()
                        refresh++
                    }
                }) { Text("Reset") }
            },
            dismissButton = { TextButton(onClick = { confirmReset = false }) { Text("Cancel") } }
        )
    }
}

@Composable
private fun TrainingRunItem(run: TrainingRunRecord) {
    val context = LocalContext.current
    val date = DateFormat.getMediumDateFormat(context).format(Date(run.timeStamp * 1000)) + " " +
            DateFormat.getTimeFormat(context).format(Date(run.timeStamp * 1000))
    val quality = if (run.heldOutBefore != null && run.heldOutAfter != null) {
        // Convert average log-probability change into "% more likely" for the held-out words
        val ratio = kotlin.math.exp((run.heldOutAfter - run.heldOutBefore).toDouble())
        val pct = ((ratio - 1.0) * 100).roundToInt()
        if (pct >= 0) "test words $pct% more likely" else "test words ${-pct}% less likely"
    } else "quality not measured"
    val status = if (run.accepted) "kept" else "discarded"

    SettingItem(
        title = "$date — $status",
        subtitle = "${run.historyEntries} words, ${run.steps} steps, ${"%.1f".format(run.minutes)} min · $quality" +
                if (run.note.isNotBlank()) "\n${run.note}" else ""
    ) {}
}
