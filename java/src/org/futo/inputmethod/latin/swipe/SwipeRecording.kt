package org.futo.inputmethod.latin.swipe

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.datastore.preferences.core.booleanPreferencesKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.json.Json
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.futo.inputmethod.latin.xlm.personal.EncryptedFile
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.GZIPOutputStream

/** "Record my swipes for training" (Stage 1 of the swipe plan). Off by default. */
val SwipeRecordingEnabled = SettingsKey(booleanPreferencesKey("swipe_recording_enabled"), false)

/**
 * Stores your labelled swipes on the phone, encrypted, until you export them to train a swipe
 * model on your PC. Nothing is uploaded anywhere.
 *
 * Records are kept in memory and written in small encrypted chunks (so a swipe never rewrites
 * the whole store). Export writes one gzip-compressed JSON-lines file to Downloads.
 */
object SwipeRecording {
    private const val TAG = "SwipeRecording"
    private const val FLUSH_EVERY = 20
    private val json = Json { encodeDefaults = true; ignoreUnknownKeys = true }

    private val labeler = SwipeLabeler()
    private val unsaved = ArrayList<SwipeRecord>()
    private var countLoaded = false

    private val _count = MutableStateFlow(0)
    /** Number of labelled swipes stored (saved + not yet saved). */
    val count: StateFlow<Int> = _count

    private fun dir(context: Context) = File(context.filesDir, "swipe-recordings").apply { mkdirs() }
    private fun countFile(context: Context) = File(dir(context), "count.txt")

    private fun enabled(context: Context) =
        context.isDirectBootUnlocked && context.getSetting(SwipeRecordingEnabled)

    fun ensureCountLoaded(context: Context) = synchronized(this) {
        if (countLoaded) return@synchronized
        val saved = try { countFile(context).takeIf { it.isFile }?.readText()?.trim()?.toInt() } catch (e: Exception) { null }
        _count.value = (saved ?: 0) + unsaved.size
        countLoaded = true
    }

    fun onSwipe(context: Context, capture: SwipeCapture) {
        if (!enabled(context)) return
        synchronized(this) { labeler.onSwipe(capture) }
    }

    fun onRejected(context: Context) {
        if (!enabled(context)) return
        synchronized(this) { labeler.onRejected() }
    }

    fun onCommitted(context: Context, word: String) {
        if (!enabled(context)) return
        val records = synchronized(this) { labeler.onCommitted(word) }
        if (records.isEmpty()) return
        ensureCountLoaded(context)
        val flushNow = synchronized(this) {
            unsaved.addAll(records)
            _count.value += records.size
            unsaved.size >= FLUSH_EVERY
        }
        if (flushNow) {
            val app = context.applicationContext
            Thread {
                try { flush(app) } catch (e: Throwable) { Log.e(TAG, "Saving swipes failed", e) }
            }.apply { priority = Thread.MIN_PRIORITY }.start()
        }
    }

    fun onFinishInput(context: Context) {
        synchronized(this) { labeler.onFinishInput() }
        if (synchronized(this) { unsaved.isNotEmpty() }) {
            val app = context.applicationContext
            Thread {
                try { flush(app) } catch (e: Throwable) { Log.e(TAG, "Saving swipes failed", e) }
            }.apply { priority = Thread.MIN_PRIORITY }.start()
        }
    }

    /** Writes unsaved records to a new encrypted chunk. */
    fun flush(context: Context) {
        val batch = synchronized(this) {
            if (unsaved.isEmpty()) return
            ArrayList(unsaved).also { unsaved.clear() }
        }
        val text = batch.joinToString("\n") { json.encodeToString(SwipeRecord.serializer(), it) }
        val chunk = File(dir(context), "chunk_${System.currentTimeMillis()}_${batch.size}.enc")
        if (!EncryptedFile.write(chunk, text)) {
            // Put them back so they can be retried
            synchronized(this) { unsaved.addAll(0, batch) }
            return
        }
        synchronized(this) {
            val savedCount = (_count.value - unsaved.size).coerceAtLeast(0)
            try { countFile(context).writeText(savedCount.toString()) } catch (_: Exception) { }
        }
    }

    /** All stored records, decrypted (for export and the on-phone replay test). */
    fun readAll(context: Context): List<SwipeRecord> {
        flush(context)
        val out = ArrayList<SwipeRecord>()
        dir(context).listFiles { f -> f.name.startsWith("chunk_") }?.sortedBy { it.name }?.forEach { f ->
            val text = EncryptedFile.read(f) ?: return@forEach
            text.lineSequence().filter { it.isNotBlank() }.forEach { line ->
                try { out.add(json.decodeFromString(SwipeRecord.serializer(), line)) } catch (_: Exception) { }
            }
        }
        return out
    }

    fun deleteAll(context: Context) {
        synchronized(this) {
            unsaved.clear()
            labeler.onFinishInput()
            dir(context).listFiles()?.forEach { it.delete() }
            _count.value = 0
            countLoaded = true
        }
    }

    /**
     * Writes every recorded swipe to Downloads as `futo-swipes-<date>.jsonl.gz`.
     * Returns a description of where it went, or null on failure.
     */
    fun export(context: Context): String? {
        val records = readAll(context)
        if (records.isEmpty()) return null
        val name = "futo-swipes-" + SimpleDateFormat("yyyyMMdd-HHmm", Locale.US).format(Date()) + ".jsonl.gz"
        return try {
            val out = if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, "application/gzip")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val uri: Uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: return null
                context.contentResolver.openOutputStream(uri) ?: return null
            } else {
                @Suppress("DEPRECATION")
                File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), name).outputStream()
            }
            GZIPOutputStream(out).bufferedWriter().use { w ->
                records.forEach { w.write(json.encodeToString(SwipeRecord.serializer(), it)); w.write("\n") }
            }
            "Downloads/$name (${records.size} swipes)"
        } catch (e: Exception) {
            Log.e(TAG, "Export failed", e)
            null
        }
    }
}
