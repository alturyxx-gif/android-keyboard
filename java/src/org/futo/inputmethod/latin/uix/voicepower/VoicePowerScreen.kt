package org.futo.inputmethod.latin.uix.voicepower

import android.widget.Toast
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.futo.inputmethod.latin.uix.settings.ScreenTitle
import org.futo.inputmethod.latin.uix.settings.ScrollableList
import org.futo.inputmethod.latin.uix.settings.Tip
import org.futo.inputmethod.latin.uix.theme.Typography

/** Settings screen for the voice power test (custom build). */
@Composable
fun VoicePowerTestScreen(navController: NavHostController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val clipboard = LocalClipboardManager.current
    val view = LocalView.current

    var running by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("") }
    var fraction by remember { mutableFloatStateOf(0f) }
    var report by remember {
        mutableStateOf(try { VoicePowerTest.reportFile(context).takeIf { it.exists() }?.readText() } catch (e: Exception) { null })
    }

    // Keep the screen on (at a constant brightness) while measuring
    DisposableEffect(running) {
        view.keepScreenOn = running
        onDispose { view.keepScreenOn = false }
    }

    ScrollableList {
        ScreenTitle("Voice power test", showBack = !running, navController)

        Tip("Measures how much battery one transcription costs on the CPU and on the GPU, using your phone's own battery readings. It transcribes the same recording over and over on each, with idle periods in between, and takes about ${VoicePowerTest.TOTAL_SECONDS / 60 + 1} minutes.")
        Tip("For a fair result: unplug the charger, don't touch the phone, keep this screen open, and close apps that might be busy in the background (music, downloads, navigation).")

        if (!running) {
            Button(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                onClick = {
                    running = true
                    status = "Starting…"
                    fraction = 0f
                    scope.launch {
                        try {
                            report = VoicePowerTest(context).run { s, f ->
                                scope.launch(Dispatchers.Main) { status = s; fraction = f }
                            }
                        } catch (e: VoicePowerTest.TestError) {
                            withContext(Dispatchers.Main) { Toast.makeText(context, e.message, Toast.LENGTH_LONG).show() }
                            status = e.message ?: "Failed"
                        } catch (e: Exception) {
                            status = "Test failed: ${e.message}"
                        } finally {
                            running = false
                        }
                    }
                }
            ) { Text("Start test") }
        } else {
            Text(status, style = Typography.Body.MediumMl, modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp))
            LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp))
        }

        if (!running && status.isNotEmpty() && report == null) {
            Text(status, modifier = Modifier.padding(16.dp))
        }

        report?.let { r ->
            ScreenTitle("Last result")
            Tip(r)
            TextButton(
                modifier = Modifier.padding(horizontal = 16.dp),
                onClick = {
                    clipboard.setText(AnnotatedString(r))
                    Toast.makeText(context, "Copied", Toast.LENGTH_SHORT).show()
                }
            ) { Text("Copy result") }
        }
    }
}
