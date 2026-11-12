package org.futo.inputmethod.latin

import android.app.Application
import android.content.Context
import android.os.UserManager
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.datastore.preferences.core.Preferences
import androidx.work.Configuration
import org.acra.ACRA
import org.acra.config.dialog
//import org.acra.config.httpSender
//import org.acra.sender.HttpSender
import org.acra.config.mailSender
import org.acra.data.StringFormat
import org.acra.ktx.initAcra
import androidx.core.content.edit
import org.acra.builder.ReportBuilder
import org.acra.config.CoreConfigurationBuilder
import org.acra.data.CrashReportDataFactory
import org.futo.inputmethod.latin.settings.Settings
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.futo.inputmethod.latin.uix.settings.LocalDataStoreCache
import org.futo.inputmethod.latin.uix.settings.NavigationItem
import org.futo.inputmethod.latin.uix.settings.NavigationItemStyle
import org.futo.inputmethod.latin.uix.settings.pages.copyToClipboard
import kotlin.collections.plus

// Custom build: Configuration.Provider re-enabled so WorkManager (personalisation training) can
// initialise on demand - the manifest removes androidx.startup's automatic initialiser.
class CrashLoggingApplication : Application(), Configuration.Provider {
    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder().build()

    override fun attachBaseContext(base: Context?) {
        super.attachBaseContext(base)

        if(isDirectBootUnlocked) {
            try {
                if (getSharedPreferences("migrate", MODE_PRIVATE).getBoolean(
                        "wiped_work",
                        false
                    ) == false
                ) {
                    deleteDatabase("androidx.work.workdb")
                    getSharedPreferences("androidx.work.util.preferences", MODE_PRIVATE)
                        .edit { clear() }
                    getSharedPreferences("migrate", MODE_PRIVATE)
                        .edit { putBoolean("wiped_work", true) }
                }
            } catch(e: Exception) {
                e.printStackTrace()
            }
        }

        // Custom build: save every crash's details to a file the Personalisation screen can show
        // (with a Copy button), instead of asking to email FUTO about a modified build.
        installCrashFileHandler(this)
        if(BuildConfig.FLAVOR == "custom") return

        if(BuildConfig.DEBUG) return

        val userManager = getSystemService(Context.USER_SERVICE) as UserManager
        if(userManager.isUserUnlocked) {
            println("Initializing ACRA, as user is unlocked")
            initAcra {
                reportFormat = StringFormat.JSON

                dialog {
                    text = getString(
                        //if(BuildConfig.ENABLE_ACRA) {
                        //    R.string.crashed_text
                        //} else {
                        R.string.crashed_text_email
                        //}
                    )
                    title = getString(R.string.crashed_title)
                    positiveButtonText = getString(R.string.crash_report_accept)
                    negativeButtonText = getString(R.string.crash_report_reject)
                    resTheme = android.R.style.Theme_DeviceDefault_Dialog
                }


                //if(BuildConfig.ENABLE_ACRA) {
                //    httpSender {
                //        uri = BuildConfig.ACRA_URL
                //        basicAuthLogin = BuildConfig.ACRA_USER
                //        basicAuthPassword = BuildConfig.ACRA_PASSWORD
                //        httpMethod = HttpSender.Method.POST
                //    }
                //} else {
                mailSender {
                    mailTo = "keyboard@futo.org"
                    reportAsFile = true
                    reportFileName = "Crash.txt"
                    subject = "Keyboard Crash Report"
                    body =
                        "I experienced this crash. My version: ${BuildConfig.VERSION_NAME}.\n\n(Enter details here if necessary)"
                }
                //}
            }

            acraInitialized = true
        } else {
            println("Skipping ACRA, as user is locked")
        }
    }

    companion object {
        var acraInitialized = false

        fun crashFile(context: Context) = java.io.File(context.noBackupFilesDir, "last_crash.txt")

        /** Writes [t]'s details to the crash file (also used for handled training errors). */
        fun recordError(context: Context, t: Throwable, what: String) {
            try {
                val sw = java.io.StringWriter()
                t.printStackTrace(java.io.PrintWriter(sw))
                crashFile(context).writeText(
                    "$what\nVersion: ${BuildConfig.VERSION_NAME}\nTime: ${java.util.Date()}\n" +
                    "Thread: ${Thread.currentThread().name}\n\n$sw"
                )
            } catch (_: Throwable) { }
        }

        private fun installCrashFileHandler(context: Context) {
            val previous = Thread.getDefaultUncaughtExceptionHandler()
            Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
                recordError(context, throwable, "Crash in thread '${thread.name}'")
                previous?.uncaughtException(thread, throwable)
            }
        }

        fun logPreferences(preferences: Preferences) {
            if(acraInitialized) {
                preferences.asMap().forEach {
                    ACRA.errorReporter.putCustomData(it.key.name, it.value.toString())
                }
            }
        }

        @Composable fun CopyLogsOption() {
            val data = LocalDataStoreCache.current
            val context = LocalContext.current
            NavigationItem(
                title = "Copy logs",
                subtitle = "May contain sensitive data",
                style = NavigationItemStyle.MiscNoArrow,
                navigate = {
                    val json = CrashReportDataFactory(context, CoreConfigurationBuilder().build())
                        .createCrashData(ReportBuilder().message("Copy logs").customData(
                            data!!.currPreferences.asMap().map {
                                it.key.name to it.value.toString()
                            }.toMap() + mapOf("Settings" to Settings.getInstance().current.dump())
                        ))
                        .toJSON()

                    context.copyToClipboard(json)
                },
            )
        }
    }
}