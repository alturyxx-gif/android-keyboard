package org.futo.inputmethod.latin.uix.settings.pages

import android.text.format.DateUtils
import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
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
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.LinearProgressIndicator
import org.futo.inputmethod.latin.swipe.CustomSwipeModel
import org.futo.inputmethod.latin.swipe.SwipeReplay
import org.futo.inputmethod.latin.swipe.UseCustomSwipeModel
import org.futo.inputmethod.latin.uix.settings.useDataStoreValue
import org.futo.inputmethod.latin.swipe.SwipeDiagnostics
import org.futo.inputmethod.latin.swipe.SwipeRecording
import org.futo.inputmethod.latin.swipe.SwipeRecordingEnabled
import org.futo.inputmethod.latin.uix.settings.ScreenTitle
import org.futo.inputmethod.latin.uix.settings.ScrollableList
import org.futo.inputmethod.latin.uix.settings.SettingToggleDataStore
import org.futo.inputmethod.latin.uix.settings.Tip
import org.futo.inputmethod.latin.uix.theme.Typography

/** Status of the swipe engine (Stage 0 of the swipe plan). */
@Composable
fun SwipeEngineStatus() {
    val refiner by SwipeDiagnostics.refinerMatch
    val last by SwipeDiagnostics.lastSwipe

    val text = buildString {
        val l = last
        if (l == null) {
            append("No swipe yet since the keyboard started. Swipe a word, then come back here.\n\n")
        } else {
            append("Swipe model: ${l.encoder}\n")
            append("Layout refiner: ${if (l.refinerActive) "ON" else "OFF"}\n")
            append("Context model: ${if (l.contextModelActive) "ON" else "OFF"}\n")
            append("Last swipe: ${"%.1f".format(l.totalMs)} ms " +
                    "(model ${"%.1f".format(l.encoderMs)}, refiner ${"%.1f".format(l.refinerMs)}, " +
                    "word search ${"%.1f".format(l.searchMs)}, context ${"%.1f".format(l.contextMs)})\n")
            append("Top words: ${l.candidates.take(4).joinToString(", ")}\n")
            append("${DateUtils.getRelativeTimeSpanString(l.timeMillis)}\n\n")
        }
        append("Refiner check: $refiner")
        SwipeDiagnostics.customModelError.value?.let { append("\n\n$it") }
    }
    Tip(text)
}

@Composable
fun SwipeTrainingScreen(navController: NavHostController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    LaunchedEffect(Unit) { withContext(Dispatchers.IO) { SwipeRecording.ensureCountLoaded(context) } }
    val count by SwipeRecording.count.collectAsState()
    var busy by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }

    ScrollableList {
        ScreenTitle("Swipe model & training", showBack = true, navController)

        ScreenTitle("Swipe engine")
        SwipeEngineStatus()

        ScreenTitle("Swipe model")
        CustomModelSection()

        ScreenTitle("Compare on my swipes")
        ReplaySection()

        ScreenTitle("Record my swipes")
        Tip("When on, each swipe is saved together with the word you actually kept " +
                "(accepted, picked from the suggestions, or deleted and typed again). " +
                "Use Export to copy them to your PC for training a swipe model in your style. " +
                "Stored encrypted on the phone; never recorded in password or incognito fields; " +
                "nothing is uploaded.")
        SettingToggleDataStore(
            title = "Record my swipes for training",
            subtitle = "$count swipes recorded",
            setting = SwipeRecordingEnabled
        )

        Row(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(enabled = !busy && count > 0, onClick = {
                busy = true
                scope.launch {
                    val where = withContext(Dispatchers.IO) { SwipeRecording.export(context) }
                    busy = false
                    Toast.makeText(
                        context,
                        if (where != null) "Saved to $where" else "Export failed",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }) { Text("Export") }

            OutlinedButton(enabled = !busy && count > 0, onClick = { confirmDelete = true }) {
                Text("Delete all")
            }
        }
        Column(Modifier.padding(horizontal = 16.dp)) {
            Text(
                "The exported file is not encrypted. Delete it from Downloads after copying it to your PC.",
                style = Typography.Body.Medium
            )
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete recorded swipes?") },
            text = { Text("This deletes all $count recorded swipes from the phone.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    scope.launch { withContext(Dispatchers.IO) { SwipeRecording.deleteAll(context) } }
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } }
        )
    }
}


@Composable
private fun CustomModelSection() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var refresh by remember { mutableStateOf(0) }
    var busy by remember { mutableStateOf(false) }
    val installed = remember(refresh) { CustomSwipeModel.isInstalled(context) }
    val info = remember(refresh) { CustomSwipeModel.info(context) }
    val useCustom = useDataStoreValue(UseCustomSwipeModel)

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        busy = true
        scope.launch {
            val err = withContext(Dispatchers.IO) { CustomSwipeModel.import(context, uri) }
            busy = false
            refresh++
            Toast.makeText(context, err ?: "Swipe model imported", Toast.LENGTH_LONG).show()
        }
    }

    Tip(
        if (installed && info != null)
            "Custom model: ${info.displayName} (${"%.1f".format(info.sizeBytes / 1e6)} MB, ${info.config}).\n" +
                    "FUTO's layout refiner only works with FUTO's model, so it's off while yours is used."
        else
            "Import a model trained on your PC with swipe-training/ (the swipe-model-*.zip from export.py). " +
                    "It's test-run in the swipe engine before it's accepted."
    )
    if (installed) {
        SettingToggleDataStore(
            title = "Use my custom swipe model",
            subtitle = if (useCustom) "On: your model is used for swiping" else "Off: FUTO's model is used",
            setting = UseCustomSwipeModel
        )
    }
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Button(enabled = !busy, onClick = { picker.launch(arrayOf("application/zip", "application/octet-stream", "*/*")) }) {
            Text(if (installed) "Replace model" else "Import model")
        }
        if (installed) {
            OutlinedButton(enabled = !busy, onClick = {
                CustomSwipeModel.delete(context)
                refresh++
            }) { Text("Remove") }
        }
    }
}

@Composable
private fun ReplaySection() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var progress by remember { mutableStateOf(0f) }
    var result by remember { mutableStateOf<String?>(null) }

    Tip("Replays the newest 15% of your recorded swipes (the part fine-tuning never trains on) " +
            "through FUTO's model and your custom model, using the keyboard's own dictionary.")
    Button(
        modifier = Modifier.padding(horizontal = 16.dp),
        enabled = !running,
        onClick = {
            running = true
            progress = 0f
            result = null
            scope.launch {
                result = withContext(Dispatchers.Default) {
                    try {
                        SwipeReplay.run(context) { progress = it }.joinToString("\n") { it.line() }
                    } catch (e: IllegalStateException) {
                        e.message
                    } catch (e: Throwable) {
                        "Comparison failed: ${e.message}"
                    }
                }
                running = false
            }
        }
    ) { Text("Compare models") }
    if (running) {
        LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().padding(16.dp))
    }
    result?.let { Tip(it) }
}
