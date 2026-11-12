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
 *  - Each message is trained ONCE. Every word is logged with the whole text before it, so a
 *    10-word message produced 10 overlapping lines; together with several typo copies per word
 *    the model saw each message dozens of times and memorised it (first phone run: everyday
 *    English 80% worse). Lines already contained in a longer line are now dropped.
 *  - Typo-correction examples use a short context and are sampled, weighted by how the word
 *    was chosen (Stage 2.6): undone autocorrections always teach that the typed word was right.
 *  - Everyday English "replay" text is mixed in so the model doesn't drift away from normal text.
 *  - Roman Urdu sentences are excluded from English training (Stage 2.3).
 *  - Splits off held-out messages (not overlapping training text) for the quality check (2.1).
 */
object TrainingDataBuilder {
    const val MAX_CONTEXT_CHARS = 160
    const val MIN_ENTRIES = 50
    /** Words of context kept in typo-correction examples */
    const val TYPO_CONTEXT_WORDS = 4
    /** Everyday-English replay lines per personal line */
    const val REPLAY_RATIO = 1.0f
    private const val SHINGLE = 4

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

    fun plainLine(entry: HistoryLogForTraining): String =
        "${contextFor(entry)} ${entry.committedWord.trim()}".trim()

    /** Relative importance of an entry, by how the word was chosen. */
    fun weight(entry: HistoryLogForTraining): Int = when (entry.importance) {
        Importance.REVERTED_AUTOCORRECT -> 6
        Importance.KEPT_TYPED -> 5
        Importance.PICKED -> 3
        Importance.TYPED -> 2
        else -> 1 // DECIDED (autocorrect) counts least
    }

    /** Chance of adding a synthetic-typo example for an entry. */
    private fun syntheticTypoChance(entry: HistoryLogForTraining): Float = when (entry.importance) {
        Importance.PICKED -> 0.4f
        Importance.TYPED -> 0.25f
        else -> 0.15f
    }

    fun isUsable(entry: HistoryLogForTraining): Boolean {
        if (entry.committedWord.isBlank()) return false
        if (entry.romanUrdu) return false
        if (RomanUrdu.isMostlyRomanUrdu(entry.priorContext + " " + entry.committedWord)) return false
        return true
    }

    private fun shortContext(entry: HistoryLogForTraining): String =
        contextFor(entry).split(' ').filter { it.isNotBlank() }.takeLast(TYPO_CONTEXT_WORDS).joinToString(" ")

    /** Typo-correction examples for one entry (the plain text is handled per message). */
    fun typoExamplesFor(entry: HistoryLogForTraining, random: Random): List<String> {
        val ctx = shortContext(entry)
        val word = entry.committedWord.trim()
        val misspelled = entry.misspelledWord?.trim()
        val out = ArrayList<String>()
        when (entry.importance) {
            Importance.REVERTED_AUTOCORRECT, Importance.KEPT_TYPED -> {
                // The user insisted on what they typed: teach "typed -> typed"
                val typed = misspelled ?: word
                repeat(2) { out.add(TrainingDataGenerator.concatFormatWordMisspelling(ctx, typed, word)) }
            }
            else -> {
                if (misspelled != null && misspelled != word) {
                    // A real typo -> correction pair from the user's own typing
                    out.add(TrainingDataGenerator.concatFormatWordMisspelling(ctx, misspelled, word))
                }
                if (random.nextFloat() < syntheticTypoChance(entry)) {
                    val correctness = listOf(4.0f, 1.0f, 0.8f, 0.6f)[random.nextInt(4)]
                    out.add(TrainingDataGenerator.concatWordMisspelling(ctx, word, correctness))
                }
            }
        }
        return out.map { it.trim() }.filter { it.isNotBlank() }
    }

    private fun words(line: String) = line.split(' ').filter { it.isNotEmpty() }

    /** Word n-grams (n = 1..[SHINGLE]) of [line], used to detect text contained in other text. */
    private fun addShingles(line: String, into: MutableSet<String>) {
        val w = words(line)
        for (n in 1..SHINGLE) for (i in 0..w.size - n) into.add(w.subList(i, i + n).joinToString(" "))
    }

    private fun isCovered(line: String, shingles: Set<String>): Boolean {
        val w = words(line)
        if (w.isEmpty()) return true
        if (w.size <= SHINGLE) return shingles.contains(w.joinToString(" "))
        for (i in 0..w.size - SHINGLE) if (!shingles.contains(w.subList(i, i + SHINGLE).joinToString(" "))) return false
        return true
    }

    /**
     * Drops lines whose text is already contained in a longer line (the per-word prefixes of
     * one message), keeping one line per message.
     */
    fun distinctMessages(lines: List<String>): List<String> {
        val shingles = HashSet<String>()
        val kept = ArrayList<String>()
        for (line in lines.map { it.trim() }.filter { it.isNotEmpty() }.distinct().sortedByDescending { it.length }) {
            if (isCovered(line, shingles)) continue
            kept.add(line)
            addShingles(line, shingles)
        }
        return kept
    }

    /**
     * @param heldOutFraction fraction of messages reserved for the before/after quality check
     * @param replay everyday-English sentences mixed into training ([REPLAY_RATIO] per message)
     */
    fun build(
        entries: List<HistoryLogForTraining>,
        random: Random = Random(1234),
        heldOutFraction: Float = 0.1f,
        replay: List<String> = emptyList(),
        replayRatio: Float = REPLAY_RATIO,
    ): Result {
        val usable = entries.filter { isUsable(it) }
        if (usable.size < MIN_ENTRIES) return Result(emptyList(), emptyList(), usable.size)

        val messages = distinctMessages(usable.map { plainLine(it) }).shuffled(random)
        val nHeld = min(100, max(3, (messages.size * heldOutFraction).toInt()))
            .coerceAtMost(messages.size / 2)
        val held = messages.take(nHeld)
        val trainMessages = messages.drop(nHeld)

        // Held-out: the last few words of each held message, each scored given the text before it
        val heldLines = held.flatMap { m ->
            val w = words(m)
            (max(1, w.size - 3) until w.size).map { i -> w.subList(0, i + 1).joinToString(" ") }
        }

        // Typo examples only from entries whose text isn't part of a held-out message
        val heldShingles = HashSet<String>().also { s -> held.forEach { addShingles(it, s) } }
        val typoLines = usable.filter { !isCovered(plainLine(it), heldShingles) }
            .flatMap { typoExamplesFor(it, random) }

        val replayLines = if (replay.isEmpty()) emptyList() else {
            val n = (trainMessages.size * replayRatio).toInt()
            List(n) { replay[random.nextInt(replay.size)] }
        }

        val trainLines = (trainMessages + typoLines + replayLines).shuffled(random)
        return Result(trainLines, heldLines, usable.size)
    }
}

/** Training run limits (Stage 2.5). The step count itself is worked out natively from the
 *  number of tokens (about six passes over the de-duplicated data), see AdapterTrainer.cpp. */
object TrainingPlan {
    const val MAX_STEPS = 300 // upper limit, roughly half an hour on the phone
    const val BATCH = 4
    const val CONTEXT_TOKENS = 96
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
