package org.futo.inputmethod.latin.xlm.personal

import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.futo.inputmethod.latin.uix.isDirectBootUnlocked
import org.futo.inputmethod.latin.xlm.HistoryLogForTraining
import java.io.File

@Serializable
data class TrainingRunRecord(
    val timeStamp: Long,
    val modelName: String,
    val examples: Int,
    val historyEntries: Int,
    val steps: Int,
    val minutes: Float,
    /** Average log-probability per held-out word before / after training (higher is better). */
    val heldOutBefore: Float? = null,
    val heldOutAfter: Float? = null,
    /** Average log-probability on everyday English: plain base model vs. with the new adapter. */
    val generalBefore: Float? = null,
    val generalAfter: Float? = null,
    /** True if the new adapter was kept, false if it was rejected because it made things worse. */
    val accepted: Boolean = true,
    val note: String = "",
)

@Serializable
private data class StoreFile(
    val history: List<HistoryLogForTraining> = emptyList(),
    val urduVocab: Map<String, Int> = emptyMap(),
)

/**
 * Holds all personalisation state on the device:
 *  - the rolling typing history (encrypted, last 60 days / 20k words) - Stage 2.4
 *  - the learned Roman Urdu vocabulary - Stage 2.3
 *  - the instant-learning phrase memory (rebuilt from history) - Stage 2.7
 *  - the log of training runs, shown in the "What it learned" screen - Stage 2.1
 *
 * Everything stays in the app's private storage. History and vocabulary are encrypted with a
 * Keystore key.
 */
object PersonalisationStore {
    private const val TAG = "PersonalisationStore"
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    private val lock = Any()
    private var loaded = false
    private val history = ArrayList<HistoryLogForTraining>()
    val urduVocab = RomanUrduVocabulary()
    /** Instant-learning memory for English typing. */
    val memory = RecentPhraseMemory()
    /** Instant-learning memory for Roman Urdu typing (kept separate from English). */
    val urduMemory = RecentPhraseMemory()

    private val _historySize = MutableStateFlow(0)
    val historySize: StateFlow<Int> = _historySize

    private fun dir(context: Context) = File(context.filesDir, "personalisation").apply { mkdirs() }
    private fun storeFile(context: Context) = File(dir(context), "store.enc")
    private fun runsFile(context: Context) = File(dir(context), "training_runs.json")

    fun ensureLoaded(context: Context) {
        synchronized(lock) {
            if (loaded) return
            if (!context.isDirectBootUnlocked) return
            val text = EncryptedFile.read(storeFile(context))
            if (text != null) {
                try {
                    val data = json.decodeFromString<StoreFile>(text)
                    history.clear()
                    history.addAll(data.history)
                    urduVocab.load(data.urduVocab)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to parse personalisation store", e)
                }
            }
            migrateOldCacheLog(context)
            memory.clear()
            memory.learnFromHistory(history.filter { !it.romanUrdu })
            urduMemory.clear()
            urduMemory.learnFromHistory(history.filter { it.romanUrdu })
            _historySize.value = history.size
            loaded = true
        }
    }

    /** FUTO's original code kept un-trained history in cache/historyLog.json; import it once. */
    private fun migrateOldCacheLog(context: Context) {
        val old = File(context.cacheDir, "historyLog.json")
        if (!old.isFile) return
        try {
            val data = json.decodeFromString<List<HistoryLogForTraining>>(old.readText())
            history.addAll(data)
        } catch (e: Exception) {
            Log.w(TAG, "Could not import old history log", e)
        }
        old.delete()
    }

    fun add(context: Context, entry: HistoryLogForTraining) {
        synchronized(lock) {
            ensureLoaded(context)
            if (!loaded) return
            history.add(entry)
            val prev = entry.ngramContext.trim().split(' ').lastOrNull { it.isNotBlank() }
            if (entry.romanUrdu) {
                urduVocab.learn(entry.committedWord)
                urduMemory.learn(prev, entry.committedWord)
            } else {
                memory.learn(prev, entry.committedWord)
            }
            _historySize.value = history.size
            sinceFlush++
        }
        // Save every 50 words in the background, so little is lost if the keyboard is killed
        if (sinceFlush >= 50) {
            sinceFlush = 0
            val appContext = context.applicationContext
            Thread { flush(appContext) }.apply { priority = Thread.MIN_PRIORITY }.start()
        }
    }

    @Volatile private var sinceFlush = 0

    /** Removes the most recent entry matching [key] (used when a word is unlearned). */
    fun removeLast(context: Context, key: String): HistoryLogForTraining? {
        synchronized(lock) {
            ensureLoaded(context)
            val idx = history.indexOfLast { it.key.startsWith(key) || it.key == key }
            if (idx == -1) return null
            val removed = history.removeAt(idx)
            if (history.none { it.committedWord == removed.committedWord }) {
                memory.forget(removed.committedWord)
                urduMemory.forget(removed.committedWord)
            }
            _historySize.value = history.size
            return removed
        }
    }

    /** Removes every entry for [word] (the "delete" button in the learned-words screen). */
    fun forgetWord(context: Context, word: String) {
        synchronized(lock) {
            ensureLoaded(context)
            history.removeAll { it.committedWord.equals(word, ignoreCase = false) }
            memory.forget(word)
            urduMemory.forget(word)
            urduVocab.forget(word)
            _historySize.value = history.size
        }
        flush(context)
    }

    fun snapshot(context: Context): List<HistoryLogForTraining> = synchronized(lock) {
        ensureLoaded(context)
        ArrayList(history)
    }

    fun clearAll(context: Context) {
        synchronized(lock) {
            history.clear()
            urduVocab.clear()
            memory.clear()
            urduMemory.clear()
            _historySize.value = 0
            loaded = true
        }
        storeFile(context).delete()
        runsFile(context).delete()
    }

    /** Applies retention limits and writes everything to disk. */
    fun flush(context: Context) {
        synchronized(lock) {
            if (!loaded || !context.isDirectBootUnlocked) return
            val pruned = HistoryRetention.prune(history, System.currentTimeMillis() / 1000)
            if (pruned.size != history.size) {
                history.clear()
                history.addAll(pruned)
            }
            val text = json.encodeToString(StoreFile(ArrayList(history), HashMap(urduVocab.learnedWords())))
            EncryptedFile.write(storeFile(context), text)
            _historySize.value = history.size
        }
    }

    // ---- Training run log ----

    fun runs(context: Context): List<TrainingRunRecord> = try {
        val f = runsFile(context)
        if (f.isFile) json.decodeFromString<List<TrainingRunRecord>>(f.readText()) else emptyList()
    } catch (e: Exception) {
        emptyList()
    }

    fun addRun(context: Context, record: TrainingRunRecord) {
        val all = (runs(context) + record).takeLast(50)
        runsFile(context).writeText(json.encodeToString(all))
    }
}
