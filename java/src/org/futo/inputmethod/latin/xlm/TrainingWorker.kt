package org.futo.inputmethod.latin.xlm

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.compose.runtime.mutableStateOf
import androidx.core.app.NotificationCompat
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequest
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.withContext
import org.futo.inputmethod.latin.R
import org.futo.inputmethod.latin.uix.USE_TRANSFORMER_FINETUNING
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.futo.inputmethod.latin.xlm.personal.GeneralEnglishCheck
import org.futo.inputmethod.latin.xlm.personal.PersonalAdapterFiles
import org.futo.inputmethod.latin.xlm.personal.PersonalisationStore
import org.futo.inputmethod.latin.xlm.personal.TrainingDataBuilder
import org.futo.inputmethod.latin.xlm.personal.TrainingPlan
import org.futo.inputmethod.latin.xlm.personal.TrainingRunRecord
import java.io.File
import java.util.Locale
import java.util.concurrent.TimeUnit

/**
 * Personalisation training (Phase 2), rewritten for the custom build.
 *
 * FUTO had this disabled entirely. The new flow:
 *  1. Take the whole rolling typing history (last 60 days), not just the words since the
 *     last run (Stage 2.4)
 *  2. Build weighted training text with longer context; keep Roman Urdu out (2.3, 2.5, 2.6)
 *  3. Measure the current model on held-out words ("before")
 *  4. Train a fresh LoRA adapter, sized to the amount of data (2.5)
 *  5. Measure again with the new adapter ("after")
 *  6. Keep the new adapter only if it isn't worse; the previous one is kept for rollback (2.2)
 *  7. Record the run for the "What it learned" screen (2.1)
 *
 * The base model file is never modified.
 */
class TrainingWorker(val context: Context, val parameters: WorkerParameters) : CoroutineWorker(context, parameters) {
    private val TAG = "TrainingWorker"
    private val notificationManager =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    override suspend fun doWork(): Result {
        if(!context.getSetting(USE_TRANSFORMER_FINETUNING)) return Result.success()
        if(!context.isDirectBootUnlocked) return Result.retry()

        TrainingWorkerStatus.isTraining.value = true
        try {
            try {
                setForeground(createForegroundInfo("Learning from your typing..."))
            } catch (e: Exception) {
                // Foreground isn't allowed in some states; train anyway
                Log.w(TAG, "Could not start foreground", e)
            }
            val result = train()
            TrainingWorkerStatus.state.emit(result)
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (e: Throwable) {
            // Catch everything (incl. Errors): a crash here would make Android retry the job
            // over and over. Record it so the Personalisation screen can show it instead.
            Log.e(TAG, "Training failed", e)
            recordTrainingError(e)
            TrainingWorkerStatus.state.emit(TrainingStateWithModel(TrainingState.FatalError, null))
        } finally {
            TrainingWorkerStatus.isTraining.value = false
        }
        return Result.success()
    }

    private fun recordTrainingError(e: Throwable) {
        try {
            val sw = java.io.StringWriter()
            e.printStackTrace(java.io.PrintWriter(sw))
            java.io.File(applicationContext.noBackupFilesDir, "last_crash.txt").writeText(
                "Personalisation training failed (handled, the app did not crash)\nTime: ${java.util.Date()}\n\n$sw"
            )
            PersonalisationStore.addRun(applicationContext, TrainingRunRecord(
                timeStamp = System.currentTimeMillis() / 1000, modelName = "", examples = 0,
                historyEntries = 0, steps = 0, minutes = 0f, accepted = false,
                note = "Training failed: ${e.javaClass.simpleName}: ${e.message ?: ""}".take(300)
            ))
        } catch (_: Throwable) { }
    }

    /** Average log-probability of each held-out word given its context. Higher = better. */
    private suspend fun evaluate(model: LanguageModel, heldOut: List<String>): Float? {
        var total = 0.0
        var n = 0
        for (line in heldOut) {
            val idx = line.lastIndexOf(' ')
            if (idx <= 0) continue
            val ctx = line.substring(0, idx)
            val word = line.substring(idx + 1)
            val score = model.scoreContinuations(ctx, listOf(word))?.firstOrNull() ?: return null
            if (!score.isFinite()) continue
            total += score
            n++
        }
        return if (n == 0) null else (total / n).toFloat()
    }

    private data class Quality(val personal: Float?, val general: Float?)

    private suspend fun evaluateWith(modelInfo: ModelInfoLoader, locale: Locale, heldOut: List<String>, adapter: File?): Quality {
        val lm = LanguageModel(
            applicationContext, null, modelInfo, locale,
            adapterOverride = adapter,
            forceNoAdapter = adapter == null
        )
        return try {
            Quality(
                personal = evaluate(lm, heldOut),
                general = if (locale.language == "en") {
                    evaluate(lm, GeneralEnglishCheck.LINES.map { it.replace('|', ' ') })
                } else null
            )
        } catch (e: Exception) {
            Log.e(TAG, "Evaluation failed", e)
            Quality(null, null)
        } finally {
            lm.closeInternalLocked()
        }
    }

    private suspend fun train(): TrainingStateWithModel {
        PersonalisationStore.ensureLoaded(applicationContext)
        PersonalisationStore.flush(applicationContext)
        val history = PersonalisationStore.snapshot(applicationContext)

        val options = ModelPaths.getModelOptions(applicationContext)
        var lastResult = TrainingStateWithModel(TrainingState.ErrorInadequateData, null)

        for ((language, modelInfo) in options) {
            val entries = history.filter { it.locale == language }
            val data = TrainingDataBuilder.build(entries)
            if (data.train.isEmpty()) continue

            val modelFile = modelInfo.path
            val locale = Locale.forLanguageTag(language)
            TrainingWorkerStatus.state.emit(TrainingStateWithModel(TrainingState.Training, modelFile.nameWithoutExtension))

            val started = System.currentTimeMillis()
            val currentAdapter = PersonalAdapterFiles.activeAdapter(modelFile)
            val before = evaluateWith(modelInfo, locale, data.heldOut, currentAdapter)
            val baseGeneral = if (currentAdapter == null) before.general
                else evaluateWith(modelInfo, locale, emptyList(), null).general

            val pending = PersonalAdapterFiles.pendingFor(modelFile)
            pending.delete()

            val steps = TrainingPlan.steps(data.train.size)
            val builder = AdapterTrainerBuilder(modelFile.absolutePath, pending.absolutePath, "")
            builder.setLossFlow(TrainingWorkerStatus.loss)
            builder.setProgressFlow(TrainingWorkerStatus.progress)
            builder.addExamples(data.train)

            val trainer = try {
                builder.loadAndPrepare()
            } catch (e: InadequateDataException) {
                continue
            }

            val threads = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(2, 6)
            trainer.setPlan(steps, TrainingPlan.CONTEXT_TOKENS, TrainingPlan.BATCH, threads)

            val powerManager = applicationContext.getSystemService(Context.POWER_SERVICE) as PowerManager
            val wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "FUTOLatinIME::modelTrainer")
            val ok = withContext(Dispatchers.Default) {
                wakeLock.acquire(120 * 60 * 1000L)
                try {
                    trainer.train()
                } finally {
                    wakeLock.release()
                    trainer.close()
                }
            }

            if (!ok || !pending.isFile) {
                PersonalisationStore.addRun(applicationContext, TrainingRunRecord(
                    timeStamp = System.currentTimeMillis() / 1000, modelName = modelFile.nameWithoutExtension,
                    examples = data.train.size, historyEntries = data.usedEntries, steps = steps,
                    minutes = (System.currentTimeMillis() - started) / 60000f,
                    heldOutBefore = before.personal, accepted = false, note = "Training failed"
                ))
                lastResult = TrainingStateWithModel(TrainingState.FatalError, modelFile.nameWithoutExtension)
                continue
            }

            val after = evaluateWith(modelInfo, locale, data.heldOut, pending)

            // Keep the new adapter only if (a) your own held-out typing didn't get worse and
            // (b) everyday English didn't get noticeably worse compared to the plain base model
            val personalOk = before.personal == null || after.personal == null || after.personal >= before.personal - 0.05f
            val generalOk = baseGeneral == null || after.general == null ||
                    after.general >= baseGeneral - GeneralEnglishCheck.MAX_ALLOWED_DROP
            val accepted = personalOk && generalOk
            if (accepted) {
                PersonalAdapterFiles.promote(modelFile)
            } else {
                pending.delete()
            }

            PersonalisationStore.addRun(applicationContext, TrainingRunRecord(
                timeStamp = System.currentTimeMillis() / 1000,
                modelName = modelFile.nameWithoutExtension,
                examples = data.train.size,
                historyEntries = data.usedEntries,
                steps = steps,
                minutes = (System.currentTimeMillis() - started) / 60000f,
                heldOutBefore = before.personal,
                heldOutAfter = after.personal,
                generalBefore = baseGeneral,
                generalAfter = after.general,
                accepted = accepted,
                note = when {
                    accepted -> ""
                    !personalOk -> "New training made your predictions worse, kept the previous version"
                    else -> "New training hurt general English too much, kept the previous version"
                }
            ))

            ModelPaths.signalReloadModels()
            lastResult = TrainingStateWithModel(TrainingState.Finished, modelFile.nameWithoutExtension)
        }

        return lastResult
    }

    private fun createForegroundInfo(progress: String): ForegroundInfo {
        val title = "Keyboard personalisation"
        val cancel = "Stop"
        val intent = workManager(applicationContext).createCancelPendingIntent(id)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            createChannel()
        }

        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle(title)
            .setTicker(title)
            .setContentText(progress)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_delete, cancel, intent)
            .build()

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            ForegroundInfo(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            ForegroundInfo(NOTIFICATION_ID, notification)
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private fun createChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Model Training Notifications",
            NotificationManager.IMPORTANCE_MIN
        )
        notificationManager.createNotificationChannel(channel)
    }
}

/**
 * FUTO's manifest removes androidx.startup's InitializationProvider (tools:node="remove"), so
 * WorkManager is never initialised automatically and WorkManager.getInstance() throws. Initialise
 * it on first use instead. (Without this, "Train now" crashed and background training was never
 * scheduled.)
 */
private val workManagerLock = Any()
fun workManager(context: Context): WorkManager = synchronized(workManagerLock) {
    try {
        WorkManager.getInstance(context)
    } catch (e: IllegalStateException) {
        try {
            WorkManager.initialize(context.applicationContext, androidx.work.Configuration.Builder().build())
        } catch (_: IllegalStateException) {
            // Already initialised by another thread/process in the meantime
        }
        WorkManager.getInstance(context)
    }
}

private const val WORKER_TAG: String = "TRAINING_WORKER"
private const val PERIODIC_NAME = "personalisation_training"
private const val IMMEDIATE_NAME = "personalisation_training_now"

/**
 * Trains roughly twice a day while the phone is charging and the battery isn't low.
 * (FUTO's version also required "device idle", which on many phones almost never happens.)
 */
fun scheduleTrainingWorkerBackground(context: Context) {
    if(!context.isDirectBootUnlocked) return

    val constraints = Constraints.Builder()
        .setRequiresBatteryNotLow(true)
        .setRequiresCharging(true)
        .build()

    val request = PeriodicWorkRequest.Builder(
        TrainingWorker::class.java,
        12L, TimeUnit.HOURS,
    ).addTag(WORKER_TAG).setConstraints(constraints).build()

    workManager(context).enqueueUniquePeriodicWork(
        PERIODIC_NAME, ExistingPeriodicWorkPolicy.KEEP, request
    )
}

/** "Train now" button. */
fun scheduleTrainingWorkerImmediately(context: Context, model: ModelInfoLoader? = null, trainingData: String? = null) {
    val workRequest = OneTimeWorkRequestBuilder<TrainingWorker>()
        .setInitialDelay(0, TimeUnit.SECONDS)
        .addTag(WORKER_TAG)
        .build()

    workManager(context).enqueueUniqueWork(IMMEDIATE_NAME, ExistingWorkPolicy.KEEP, workRequest)
}

val NUM_TRAINING_RUNS_KEY = intPreferencesKey("training_runs_count")

const val CHANNEL_ID = "TRAINING"
const val NOTIFICATION_ID = 1

enum class TrainingState {
    None,
    Training,
    ErrorInadequateData,
    Finished,
    FatalError,
}

data class TrainingStateWithModel(
    val state: TrainingState,
    val model: String?
)

enum class LanguageModelFacilitatorRequest {
    ResetModel,
    ClearTrainingLog
}

object TrainingWorkerStatus {
    val state = MutableSharedFlow<TrainingStateWithModel>(replay = 1)
    val lmRequest = MutableSharedFlow<LanguageModelFacilitatorRequest>(replay = 0)
    val isTraining = mutableStateOf(false)

    val loss = MutableSharedFlow<Float>(replay = 4)
    val progress = MutableSharedFlow<Float>(replay = 4)
}
