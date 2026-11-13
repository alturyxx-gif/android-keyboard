package org.futo.inputmethod.latin.tap

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.datastore.preferences.core.booleanPreferencesKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.json.Json
import org.futo.inputmethod.keyboard.Keyboard
import org.futo.inputmethod.keyboard.KeyboardSwitcher
import org.futo.inputmethod.latin.uix.SettingsKey
import org.futo.inputmethod.latin.uix.getSetting
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.futo.inputmethod.latin.xlm.personal.EncryptedFile
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.GZIPOutputStream

/** "Record my taps for tuning" (Tap plan, Stage 0.1). Off by default. */
val TapRecordingEnabled = SettingsKey(booleanPreferencesKey("tap_recording_enabled"), false)

/**
 * Hooks called from the input logic for every tapped letter, backspace and committed word.
 * Always updates [TapStats] (numbers only). When recording is on, also keeps labelled taps,
 * encrypted on the phone, for replay testing and tuning (never in private fields).
 */
object TapRecording {
    private const val TAG = "TapRecording"
    private const val FLUSH_EVERY = 40
    private val json = Json { encodeDefaults = true; ignoreUnknownKeys = true }

    private val labeler = TapLabeler()
    private val unsaved = ArrayList<TapRecord>()
    private var countLoaded = false
    private val _count = MutableStateFlow(0)
    val count: StateFlow<Int> = _count

    private var cachedKeyboard: Keyboard? = null
    private var cachedBoxes: KeyBoxes? = null

    private fun dir(context: Context) = File(context.filesDir, "tap-recordings").apply { mkdirs() }
    private fun countFile(context: Context) = File(dir(context), "count.txt")

    private fun recording(context: Context) =
        context.isDirectBootUnlocked && context.getSetting(TapRecordingEnabled)

    /** Letter keys of the current keyboard (cached per keyboard). */
    @JvmStatic
    fun currentKeys(): KeyBoxes? = synchronized(this) {
        val kb = KeyboardSwitcher.getInstance()?.keyboard ?: return null
        if (kb === cachedKeyboard) return cachedBoxes
        val keys = kb.sortedKeys.filter { Character.isLetter(it.code) }
            .distinctBy { Character.toLowerCase(it.code) }
        cachedKeyboard = kb
        val sp = kb.sortedKeys.firstOrNull { it.code == ' '.code }
        cachedBoxes = if (keys.size < 10) null else KeyBoxes(
            width = kb.mBaseWidth, height = kb.mBaseHeight,
            letters = keys.joinToString("") { Character.toString(Character.toLowerCase(it.code)) },
            x = keys.map { it.x }, y = keys.map { it.y }, w = keys.map { it.width }, h = keys.map { it.height },
            space = sp?.let { listOf(it.x, it.y, it.width, it.height) } ?: emptyList(),
        )
        cachedBoxes
    }

    private fun words(prev: String?): List<String> =
        prev?.split(' ')?.filter { it.isNotBlank() && it != "<S>" }?.takeLast(10) ?: emptyList()

    // ---- hooks (called from InputLogic) ----

    @JvmStatic
    fun onLetter(context: Context, codePoint: Int, x: Int, y: Int, index: Int, prevWords: String?, isPrivate: Boolean) {
        TapStats.onLetter(context)
        TapKeyFeedback.onLetterTyped(context)
        if (isPrivate || !Character.isLetter(codePoint)) return
        val fix = synchronized(this) {
            labeler.onLetterTyped(codePoint.toChar(), x, y, index, System.currentTimeMillis(), words(prevWords), currentKeys())
        } ?: return
        TapStats.onMidWordFix(context)
        currentKeys()?.let { TapPredictor.learnFix(context, fix.word[0], fix.tapX[0], fix.tapY[0], it) }
        if (recording(context)) add(context, listOf(fix))
    }

    /** Space typed while a word was being composed (index = letters in the word). */
    @JvmStatic
    fun onSpace(context: Context, index: Int, isPrivate: Boolean) {
        TapKeyFeedback.onSpaceTyped(context)
        if (isPrivate) return
        val fix = synchronized(this) { labeler.onSpaceTyped(index, System.currentTimeMillis(), emptyList(), currentKeys()) } ?: return
        TapStats.onMidWordFix(context)
        if (recording(context)) add(context, listOf(fix))
    }

    @JvmStatic
    fun onBackspace(context: Context, composing: Boolean, deletedCodePoint: Int, x: Int, y: Int, index: Int, prefix: String) {
        TapStats.onBackspace(context)
        TapKeyFeedback.onBackspace(context)
        synchronized(this) {
            if (composing) {
                if (deletedCodePoint > 0 && x >= 0) labeler.onLetterDeleted(deletedCodePoint.toChar(), x, y, index, prefix)
            } else {
                labeler.onEditCommitted()
            }
        }
    }

    @JvmStatic
    fun onCommit(context: Context, typed: String, committed: String, manualPick: Boolean, isBatch: Boolean,
                 tapX: IntArray, tapY: IntArray, prevWords: String?, isPrivate: Boolean) {
        if (isBatch) return // swipes are recorded separately
        TapStats.onWord(context, autocorrected = !manualPick && !committed.equals(typed, ignoreCase = true))
        if (isPrivate) { synchronized(this) { labeler.onEditCommitted() }; return }
        val keys = currentKeys() ?: return
        // Stage 2: learn where you tap from letters that went in as meant
        if (committed.length == typed.length && typed.length == tapX.size) {
            TapPredictor.learnTouch(context, committed, tapX, tapY, keys, typed)
        }
        val out = synchronized(this) {
            labeler.onCommit(TapCommit(System.currentTimeMillis(), typed, committed, manualPick,
                tapX, tapY, words(prevWords), keys))
        }
        if (recording(context)) add(context, out)
    }

    @JvmStatic
    fun onRevert(context: Context) {
        TapStats.onReverted(context)
        val out = synchronized(this) { labeler.onRevert() }
        if (recording(context)) add(context, out)
    }

    @JvmStatic
    fun onFinishInput(context: Context) {
        val out = synchronized(this) { labeler.onFinishInput() }
        if (recording(context)) add(context, out)
        TapStats.save(context)
        if (synchronized(this) { unsaved.isNotEmpty() }) flushInBackground(context)
    }

    // ---- storage ----

    fun ensureCountLoaded(context: Context) = synchronized(this) {
        if (countLoaded) return@synchronized
        val saved = try { countFile(context).takeIf { it.isFile }?.readText()?.trim()?.toInt() } catch (e: Exception) { null }
        _count.value = (saved ?: 0) + unsaved.size
        countLoaded = true
    }

    private fun add(context: Context, records: List<TapRecord>) {
        if (records.isEmpty()) return
        ensureCountLoaded(context)
        val flushNow = synchronized(this) {
            unsaved.addAll(records)
            _count.value += records.size
            unsaved.size >= FLUSH_EVERY
        }
        if (flushNow) flushInBackground(context)
    }

    private fun flushInBackground(context: Context) {
        val app = context.applicationContext
        Thread {
            try { flush(app) } catch (e: Throwable) { Log.e(TAG, "Saving taps failed", e) }
        }.apply { priority = Thread.MIN_PRIORITY }.start()
    }

    fun flush(context: Context) {
        val batch = synchronized(this) {
            if (unsaved.isEmpty()) return
            ArrayList(unsaved).also { unsaved.clear() }
        }
        val text = batch.joinToString("\n") { json.encodeToString(TapRecord.serializer(), it) }
        val chunk = File(dir(context), "chunk_${System.currentTimeMillis()}_${batch.size}.enc")
        if (!EncryptedFile.write(chunk, text)) {
            synchronized(this) { unsaved.addAll(0, batch) }
            return
        }
        synchronized(this) {
            val saved = (_count.value - unsaved.size).coerceAtLeast(0)
            try { countFile(context).writeText(saved.toString()) } catch (_: Exception) { }
        }
    }

    fun readAll(context: Context): List<TapRecord> {
        flush(context)
        val out = ArrayList<TapRecord>()
        dir(context).listFiles { f -> f.name.startsWith("chunk_") }?.sortedBy { it.name }?.forEach { f ->
            val text = EncryptedFile.read(f) ?: return@forEach
            text.lineSequence().filter { it.isNotBlank() }.forEach { line ->
                try { out.add(json.decodeFromString(TapRecord.serializer(), line)) } catch (_: Exception) { }
            }
        }
        return out
    }

    fun deleteAll(context: Context) = synchronized(this) {
        unsaved.clear()
        labeler.onFinishInput()
        dir(context).listFiles()?.forEach { it.delete() }
        _count.value = 0
        countLoaded = true
    }

    /** Writes all recorded taps to Downloads as futo-taps-<date>.jsonl.gz. */
    fun export(context: Context): String? {
        val records = readAll(context)
        if (records.isEmpty()) return null
        val name = "futo-taps-" + SimpleDateFormat("yyyyMMdd-HHmm", Locale.US).format(Date()) + ".jsonl.gz"
        return try {
            val out = if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, name)
                    put(MediaStore.Downloads.MIME_TYPE, "application/gzip")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: return null
                context.contentResolver.openOutputStream(uri) ?: return null
            } else {
                @Suppress("DEPRECATION")
                File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), name).outputStream()
            }
            GZIPOutputStream(out).bufferedWriter().use { w ->
                records.forEach { w.write(json.encodeToString(TapRecord.serializer(), it)); w.write("\n") }
            }
            "Downloads/$name (${records.size} records)"
        } catch (e: Exception) {
            Log.e(TAG, "Export failed", e)
            null
        }
    }
}
