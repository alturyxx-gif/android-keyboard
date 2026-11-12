package org.futo.inputmethod.latin.xlm.personal

import org.futo.inputmethod.latin.xlm.HistoryLogForTraining
import org.futo.inputmethod.latin.xlm.TrainingDataGenerator
import org.futo.inputmethod.latin.xlm.grammar.RomanUrdu
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * Pure (Android-free) personalisation logic, kept separate so it can be unit tested.
 *
 * Importance values recorded with each history entry:
 *  -1 = typed word committed as-is (user typed it, no correction offered or taken)
 *   0 = keyboard decided (autocorrect, or the typed word when no correction applied)
 *   1 = user tapped a suggestion
 *   3 = user tapped their own typed word in the strip (explicitly rejected autocorrect)
 *   4 = user undid an autocorrection with backspace (the typed word was intended)
 */
object Importance {
    const val TYPED = -1
    const val DECIDED = 0
    const val PICKED = 1
    const val KEPT_TYPED = 3
    const val REVERTED_AUTOCORRECT = 4
}

object HistoryRetention {
    const val MAX_AGE_DAYS = 60
    const val MAX_ENTRIES = 20_000

    /** Drops entries older than [maxAgeDays] and keeps at most [maxEntries] newest ones. */
    fun prune(
        entries: List<HistoryLogForTraining>,
        nowSeconds: Long,
        maxAgeDays: Int = MAX_AGE_DAYS,
        maxEntries: Int = MAX_ENTRIES,
    ): List<HistoryLogForTraining> {
        val cutoff = nowSeconds - maxAgeDays.toLong() * 24 * 3600
        val recent = entries.filter { it.timeStamp >= cutoff }.sortedBy { it.timeStamp }
        return if (recent.size > maxEntries) recent.takeLast(maxEntries) else recent
    }
}

/**
 * Turns the rolling typing history into training text for the LoRA adapter.
 *
 * Changes vs. the original FUTO logic:
 *  - Uses the longer `priorContext` (up to [MAX_CONTEXT_CHARS]) instead of just the last few
 *    words, so the model learns phrasing (Stage 2.5).
 *  - Weights examples by how the word was chosen (Stage 2.6): words the user picked or typed
 *    deliberately count more than words autocorrect inserted, and undone autocorrections teach
 *    the model that the typed word was right.
 *  - Roman Urdu sentences are excluded from English training (Stage 2.3).
 *  - Splits off a held-out set so each training run can be checked for quality (Stage 2.1).
 */
object TrainingDataBuilder {
    const val MAX_CONTEXT_CHARS = 160
    const val MIN_ENTRIES = 50

    data class Result(val train: List<String>, val heldOut: List<String>, val usedEntries: Int)

    fun contextFor(entry: HistoryLogForTraining): String {
        val prior = entry.priorContext.ifBlank { entry.ngramContext }
        val trimmed = prior.trim()
        if (trimmed.length <= MAX_CONTEXT_CHARS) return trimmed
        // Cut at a word boundary
        val cut = trimmed.substring(trimmed.length - MAX_CONTEXT_CHARS)
        val space = cut.indexOf(' ')
        return if (space in 0 until cut.length - 1) cut.substring(space + 1) else cut
    }

    /** Number of training copies for an entry, by how the word was chosen. */
    fun weight(entry: HistoryLogForTraining): Int = when (entry.importance) {
        Importance.REVERTED_AUTOCORRECT -> 6
        Importance.KEPT_TYPED -> 5
        Importance.PICKED -> 3
        Importance.TYPED -> 2
        else -> 1 // DECIDED (autocorrect) counts least
    }

    fun isUsable(entry: HistoryLogForTraining): Boolean {
        if (entry.committedWord.isBlank()) return false
        if (entry.romanUrdu) return false
        if (RomanUrdu.isMostlyRomanUrdu(entry.priorContext + " " + entry.committedWord)) return false
        return true
    }

    fun examplesFor(entry: HistoryLogForTraining, random: Random): List<String> {
        val ctx = contextFor(entry)
        val word = entry.committedWord.trim()
        val misspelled = entry.misspelledWord?.trim()
        val n = weight(entry)
        val out = ArrayList<String>()

        when (entry.importance) {
            Importance.REVERTED_AUTOCORRECT, Importance.KEPT_TYPED -> {
                // The user insisted on what they typed: teach "typed -> typed"
                val typed = misspelled ?: word
                repeat(n) { out.add(TrainingDataGenerator.concatFormatWordMisspelling(ctx, typed, word)) }
                out.add("$ctx $word")
            }
            else -> {
                // Plain next-word example (prediction)
                repeat(max(1, n / 2)) { out.add("$ctx $word") }
                if (misspelled != null && misspelled != word) {
                    // A real typo -> correction pair from the user's own typing
                    repeat(n) { out.add(TrainingDataGenerator.concatFormatWordMisspelling(ctx, misspelled, word)) }
                }
                // Synthetic typos so the model learns to correct towards this word
                repeat(n) {
                    val correctness = listOf(4.0f, 1.0f, 0.8f, 0.6f)[random.nextInt(4)]
                    out.add(TrainingDataGenerator.concatWordMisspelling(ctx, word, correctness))
                }
            }
        }
        return out.map { it.trim() }.filter { it.isNotBlank() }
    }

    /**
     * @param heldOutFraction fraction of entries reserved for the before/after quality check
     */
    fun build(
        entries: List<HistoryLogForTraining>,
        random: Random = Random(1234),
        heldOutFraction: Float = 0.1f,
    ): Result {
        val usable = entries.filter { isUsable(it) }
        if (usable.size < MIN_ENTRIES) return Result(emptyList(), emptyList(), usable.size)

        val shuffled = usable.shuffled(random)
        val nHeld = min(500, max(10, (shuffled.size * heldOutFraction).toInt()))
        val held = shuffled.take(nHeld)
        val train = shuffled.drop(nHeld)

        val trainLines = train.flatMap { examplesFor(it, random) }.shuffled(random)
        // Held-out: plain "context word" lines, scored by the model before/after training
        val heldLines = held.map { "${contextFor(it)} ${it.committedWord.trim()}".trim() }
        return Result(trainLines, heldLines, usable.size)
    }
}

/**
 * How big each training run should be (Stage 2.5). The original code always ran 128 steps
 * regardless of data size; this scales with the amount of data so more history is actually used.
 */
object TrainingPlan {
    const val MIN_STEPS = 32
    const val MAX_STEPS = 600
    const val BATCH = 4
    const val CONTEXT_TOKENS = 128

    /**
     * About one pass over the training lines. Host tests on the real model showed that many
     * passes over a small history makes the model memorise it and get worse at general English,
     * so this deliberately stays close to a single pass.
     */
    fun steps(numTrainingLines: Int): Int {
        val wanted = numTrainingLines / BATCH
        return wanted.coerceIn(MIN_STEPS, MAX_STEPS)
    }
}

/**
 * Everyday English used to check that a new adapter hasn't made the model worse at normal
 * text (Stage 2.1 quality check). Each entry is "context|next word".
 */
object GeneralEnglishCheck {
    val LINES: List<String> = listOf(
        "I am going to the|store", "Can you send me the|link", "What time does the|meeting",
        "Thank you so much for your|help", "I will be there in ten|minutes", "Let me know if you|need",
        "Do you want to grab|lunch", "I think we should|go", "How was your|day",
        "See you|tomorrow", "Sorry I missed your|call", "Happy|birthday",
        "I don't know what to|do", "It was really nice to|meet", "Could you please|send",
        "I'm on my|way", "Have a good|night", "That sounds like a good|idea",
        "Did you see the|game", "I need to go to the|doctor", "We should get|together",
        "Can we talk|later", "I just got|home", "The weather is really|nice",
        "Are you free this|weekend", "I'll call you when I|get", "Please let me|know",
        "I forgot my|phone", "What do you want for|dinner", "Good|morning",
    )

    /** Reject a new adapter if general English got worse by more than this (avg nats/word). */
    const val MAX_ALLOWED_DROP = 0.35f
}

/**
 * Instant learning (Stage 2.7): remembers words and word pairs from recent typing so that
 * new words and names show up in suggestions straight away, without waiting for overnight
 * training.
 */
class RecentPhraseMemory(private val maxPairs: Int = 5000) {
    private val unigrams = HashMap<String, Int>()
    private val pairs = LinkedHashMap<String, HashMap<String, Int>>(16, 0.75f, true)

    fun clear() {
        unigrams.clear()
        pairs.clear()
    }

    fun learn(previousWord: String?, word: String) {
        val w = word.trim()
        if (w.isEmpty() || w.any { it.isWhitespace() }) return
        unigrams[w] = (unigrams[w] ?: 0) + 1
        val p = previousWord?.trim()?.lowercase()
        if (!p.isNullOrEmpty()) {
            val m = pairs.getOrPut(p) { HashMap() }
            m[w] = (m[w] ?: 0) + 1
            if (pairs.size > maxPairs) {
                val eldest = pairs.keys.iterator().next()
                pairs.remove(eldest)
            }
        }
    }

    fun forget(word: String) {
        val w = word.trim()
        unigrams.remove(w)
        pairs.values.forEach { it.remove(w) }
    }

    /** Words most often typed after [previousWord], best first. */
    fun nextWords(previousWord: String, limit: Int = 3, minCount: Int = 2): List<Pair<String, Int>> =
        pairs[previousWord.trim().lowercase()]
            ?.filter { it.value >= minCount }
            ?.entries?.sortedByDescending { it.value }
            ?.take(limit)?.map { it.key to it.value }
            ?: emptyList()

    /** Previously typed words that complete [prefix], best first. */
    fun completions(prefix: String, limit: Int = 3, minCount: Int = 2): List<Pair<String, Int>> {
        if (prefix.isEmpty()) return emptyList()
        return unigrams.entries
            .filter { it.value >= minCount && it.key.length > prefix.length && it.key.startsWith(prefix, ignoreCase = true) }
            .sortedByDescending { it.value }
            .take(limit).map { it.key to it.value }
    }

    fun count(word: String): Int = unigrams[word.trim()] ?: 0

    fun learnFromHistory(entries: List<HistoryLogForTraining>) {
        for (e in entries) {
            val prev = e.ngramContext.trim().split(' ').lastOrNull { it.isNotBlank() }
            learn(prev, e.committedWord)
        }
    }
}

/**
 * Roman Urdu vocabulary (Stage 2.3): the built-in common-word list plus words the user has
 * typed in Roman Urdu sentences. Words in here are protected from English autocorrect when
 * typed in a Roman Urdu sentence, and are offered as completions there.
 */
class RomanUrduVocabulary {
    private val learned = HashMap<String, Int>()

    fun learnedWords(): Map<String, Int> = learned

    fun load(words: Map<String, Int>) {
        learned.clear()
        learned.putAll(words)
    }

    fun learn(word: String) {
        val w = word.trim().lowercase()
        if (w.isEmpty() || !w.all { it.isLetter() || it == '\'' }) return
        learned[w] = (learned[w] ?: 0) + 1
    }

    fun forget(word: String) {
        learned.remove(word.trim().lowercase())
    }

    fun clear() = learned.clear()

    fun knows(word: String): Boolean {
        val w = word.trim().lowercase()
        return RomanUrdu.isRomanUrduWord(w) || (learned[w] ?: 0) >= 2
    }

    /**
     * Should autocorrect leave [typedWord] alone?
     *  - In a Roman Urdu sentence: protect any known Roman Urdu word.
     *  - Elsewhere: protect only words the user has typed in Roman Urdu several times and that
     *    aren't English collisions (e.g. "nahi", "acha").
     */
    fun shouldProtect(typedWord: String, sentenceSoFar: String): Boolean {
        val w = typedWord.trim().lowercase()
        if (w.length < 2) return false
        if (w in RomanUrdu.ENGLISH_COLLISIONS) return false
        val inUrduSentence = RomanUrdu.isMostlyRomanUrdu("$sentenceSoFar $w")
        return if (inUrduSentence) knows(w) else (learned[w] ?: 0) >= 3
    }

    /** Completions for [prefix] from learned Roman Urdu words, best first. */
    fun completions(prefix: String, limit: Int = 3): List<String> {
        val p = prefix.trim().lowercase()
        if (p.isEmpty()) return emptyList()
        return learned.entries
            .filter { it.value >= 2 && it.key.length > p.length && it.key.startsWith(p) }
            .sortedByDescending { it.value }
            .take(limit).map { it.key }
    }
}
