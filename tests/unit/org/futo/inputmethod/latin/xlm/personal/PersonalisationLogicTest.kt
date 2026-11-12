package org.futo.inputmethod.latin.xlm.personal

import org.futo.inputmethod.latin.xlm.HistoryLogForTraining
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

private fun entry(
    word: String,
    prior: String = "I am going to the",
    importance: Int = Importance.DECIDED,
    misspelled: String? = null,
    time: Long = 1_000_000L,
    romanUrdu: Boolean = false,
) = HistoryLogForTraining(
    key = "$prior $word",
    priorContext = prior,
    ngramContext = prior.split(' ').takeLast(3).joinToString(" "),
    misspelledWord = misspelled,
    committedWord = word,
    importance = importance,
    locale = "en",
    timeStamp = time,
    romanUrdu = romanUrdu,
)

class HistoryRetentionTest {
    @Test
    fun dropsOldEntries() {
        val now = 100L * 24 * 3600
        val old = entry("old", time = now - 61L * 24 * 3600)
        val fresh = entry("fresh", time = now - 1L * 24 * 3600)
        val pruned = HistoryRetention.prune(listOf(old, fresh), now)
        assertEquals(listOf("fresh"), pruned.map { it.committedWord })
    }

    @Test
    fun capsNumberOfEntriesKeepingNewest() {
        val entries = (0 until 30).map { entry("w$it", time = 1000L + it) }
        val pruned = HistoryRetention.prune(entries, 2000L, maxAgeDays = 1, maxEntries = 10)
        assertEquals(10, pruned.size)
        assertEquals("w29", pruned.last().committedWord)
        assertEquals("w20", pruned.first().committedWord)
    }
}

class TrainingDataBuilderTest {
    @Test
    fun weightsFollowHowWordWasChosen() {
        assertTrue(TrainingDataBuilder.weight(entry("a", importance = Importance.PICKED)) >
                TrainingDataBuilder.weight(entry("a", importance = Importance.DECIDED)))
        assertTrue(TrainingDataBuilder.weight(entry("a", importance = Importance.REVERTED_AUTOCORRECT)) >
                TrainingDataBuilder.weight(entry("a", importance = Importance.PICKED)))
        assertTrue(TrainingDataBuilder.weight(entry("a", importance = Importance.KEPT_TYPED)) >
                TrainingDataBuilder.weight(entry("a", importance = Importance.TYPED)))
    }

    @Test
    fun usesLongerPriorContext() {
        val prior = "Yesterday after work I told my brother that we should probably go to the"
        val e = entry("park", prior = prior)
        assertEquals(prior, TrainingDataBuilder.contextFor(e))
        assertEquals("$prior park", TrainingDataBuilder.plainLine(e))
    }

    @Test
    fun veryLongContextIsCutAtWordBoundary() {
        val prior = (1..80).joinToString(" ") { "word$it" }
        val ctx = TrainingDataBuilder.contextFor(entry("x", prior = prior))
        assertTrue(ctx.length <= TrainingDataBuilder.MAX_CONTEXT_CHARS)
        assertTrue(ctx.startsWith("word"))
        assertTrue(ctx.endsWith("word80"))
    }

    @Test
    fun revertedAutocorrectTeachesTypedWord() {
        val e = entry("Zainab", importance = Importance.REVERTED_AUTOCORRECT, misspelled = "Zainab")
        val lines = TrainingDataBuilder.typoExamplesFor(e, Random(1))
        // Every correction example must map the typed word to itself, never to something else
        assertTrue(lines.isNotEmpty())
        assertTrue(lines.all { it.contains("Zainab") })
    }

    @Test
    fun realTypoBecomesCorrectionPair() {
        val e = entry("because", importance = Importance.DECIDED, misspelled = "becuase")
        val lines = TrainingDataBuilder.typoExamplesFor(e, Random(1))
        assertTrue(lines.any { it.contains("because") && it.contains("<XBU>") })
    }

    @Test
    fun typoExamplesUseShortContext() {
        val e = entry("because", prior = "one two three four five six seven", misspelled = "becuase")
        val line = TrainingDataBuilder.typoExamplesFor(e, Random(1)).first()
        assertTrue(line, line.startsWith("four five six seven <XBU>"))
    }

    @Test
    fun eachMessageIsTrainedOnce() {
        // Every word of one message is logged with the text before it: one line must remain
        val message = "Hey are you coming to the cafe later today"
        val w = message.split(' ')
        val lines = (1 until w.size).map { w.subList(0, it + 1).joinToString(" ") }
        assertEquals(listOf(message), TrainingDataBuilder.distinctMessages(lines))
        // Different messages are all kept
        val other = listOf("I left my keys at the office", "See you at the gym at seven")
        assertEquals(other.toSet(), TrainingDataBuilder.distinctMessages(other).toSet())
    }

    @Test
    fun prefixChainsDoNotMultiplyTrainingText() {
        // 20 messages x 10 words, logged word by word like the keyboard does
        val entries = (0 until 20).flatMap { m ->
            val w = (0 until 10).map { "m${m}w$it" }
            (1 until w.size).map { i -> entry(w[i], prior = w.subList(0, i).joinToString(" "), importance = Importance.TYPED) }
        }
        val r = TrainingDataBuilder.build(entries, Random(5))
        val plain = r.train.filter { !it.contains("<XBU>") }
        // One plain line per training message, not one per word
        assertTrue("plain lines: ${plain.size}", plain.size <= 20)
        // Typo examples are sampled, so far fewer than one per word
        val typos = r.train.count { it.contains("<XBU>") }
        assertTrue("typo lines: $typos", typos < entries.size / 2)
    }

    @Test
    fun replayIsMixedIn() {
        val entries = (0 until 100).map { entry("word$it", prior = "context number $it is") }
        val replay = listOf("Everyday sentence one.", "Everyday sentence two.")
        val r = TrainingDataBuilder.build(entries, Random(3), replay = replay, replayRatio = 1.0f)
        assertEquals(90, r.train.count { it in replay }) // one replay line per training message
    }

    @Test
    fun romanUrduEntriesExcludedFromEnglishTraining() {
        assertFalse(TrainingDataBuilder.isUsable(entry("hai", prior = "yaar mujhe nahi pata kya", romanUrdu = true)))
        assertFalse(TrainingDataBuilder.isUsable(entry("hai", prior = "yaar mujhe nahi pata kya")))
        assertTrue(TrainingDataBuilder.isUsable(entry("park", prior = "Let's go to the")))
    }

    @Test
    fun needsMinimumData() {
        val few = (0 until 10).map { entry("w$it") }
        assertTrue(TrainingDataBuilder.build(few).train.isEmpty())
    }

    @Test
    fun buildSplitsHeldOut() {
        val many = (0 until 200).map { entry("word$it", prior = "context number $it is") }
        val r = TrainingDataBuilder.build(many, Random(3))
        assertEquals(200, r.usedEntries)
        // Held-out words must not appear in any training line
        val heldWords = r.heldOut.map { it.substringAfterLast(' ') }.filter { it.startsWith("word") }.toSet()
        assertTrue(heldWords.isNotEmpty())
        assertTrue(r.train.none { line -> heldWords.any { line.endsWith(" $it") || line.contains(" $it ") } })
    }
}

class RecentPhraseMemoryTest {
    @Test
    fun learnsNextWordsAndCompletions() {
        val m = RecentPhraseMemory()
        repeat(3) { m.learn("see", "Zainab") }
        m.learn("see", "you")
        assertEquals("Zainab", m.nextWords("see").first().first)
        assertEquals(listOf("Zainab"), m.completions("Zai").map { it.first })
        // Needs at least 2 uses
        assertTrue(m.completions("yo").isEmpty())
    }

    @Test
    fun forgetRemovesWord() {
        val m = RecentPhraseMemory()
        repeat(3) { m.learn("see", "Zainab") }
        m.forget("Zainab")
        assertTrue(m.nextWords("see").isEmpty())
        assertEquals(0, m.count("Zainab"))
    }

    @Test
    fun boundedNumberOfPairs() {
        val m = RecentPhraseMemory(maxPairs = 10)
        for (i in 0 until 50) repeat(2) { m.learn("prev$i", "next") }
        assertTrue(m.nextWords("prev0").isEmpty())
        assertEquals("next", m.nextWords("prev49").first().first)
    }
}

class RomanUrduVocabularyTest {
    @Test
    fun protectsKnownWordsInUrduSentences() {
        val v = RomanUrduVocabulary()
        assertTrue(v.shouldProtect("nahi", "yaar mujhe"))
        assertTrue(v.shouldProtect("acha", "theek hai"))
    }

    @Test
    fun doesNotProtectInEnglishUnlessLearned() {
        val v = RomanUrduVocabulary()
        assertFalse(v.shouldProtect("nahi", "I want to"))
        repeat(3) { v.learn("nahi") }
        assertTrue(v.shouldProtect("nahi", "I want to"))
    }

    @Test
    fun neverProtectsEnglishCollisions() {
        val v = RomanUrduVocabulary()
        repeat(10) { v.learn("main") }
        assertFalse(v.shouldProtect("main", "yaar mujhe nahi pata"))
    }

    @Test
    fun completionsFromLearnedWords() {
        val v = RomanUrduVocabulary()
        repeat(2) { v.learn("bilkul") }
        assertEquals(listOf("bilkul"), v.completions("bil"))
        v.forget("bilkul")
        assertTrue(v.completions("bil").isEmpty())
    }
}
