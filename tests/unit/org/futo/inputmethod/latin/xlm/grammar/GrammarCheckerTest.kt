package org.futo.inputmethod.latin.xlm.grammar

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * A fake language model: every continuation starts at 0 and is penalised for each "bad"
 * phrase it contains and rewarded for each "good" phrase. Phrases are matched on the full
 * text (prefix + continuation), lowercased, so right-hand context matters just as it does
 * with the real model.
 */
class PhraseScorer(
    private val bad: Map<String, Float>,
    private val good: Map<String, Float> = emptyMap(),
) : ContinuationScorer {
    var calls = 0
    override suspend fun score(prefix: String, continuations: List<String>): FloatArray {
        calls++
        return FloatArray(continuations.size) { i ->
            val full = " " + (prefix.trimEnd() + " " + continuations[i].trim()).lowercase() + " "
            var s = 0f
            bad.forEach { (p, w) -> if (full.contains(" $p ")) s -= w }
            good.forEach { (p, w) -> if (full.contains(" $p ")) s += w }
            s
        }
    }
}

class GrammarCheckerTest {
    private fun check(checker: GrammarChecker, text: String) = runBlocking { checker.check(text) }

    @Test
    fun theirThereFixedWithHighConfidence() {
        val scorer = PhraseScorer(bad = mapOf("there car" to 6f))
        val checker = GrammarChecker(scorer)
        val text = "I like there car a lot "
        val fixes = check(checker, text)
        assertEquals(1, fixes.size)
        val fix = fixes[0]
        assertEquals("there", fix.original)
        assertEquals(FixConfidence.High, fix.confidence)
        assertEquals("there", text.substring(fix.start, fix.end))
        assertTrue(fix.replacement == "their" || fix.replacement == "they're")
    }

    @Test
    fun smallMarginIsUnderlinedNotAutoApplied() {
        val scorer = PhraseScorer(bad = mapOf("its a" to 3f, "its an" to 3f))
        val fixes = check(GrammarChecker(scorer), "I think its a good idea ")
        assertEquals(1, fixes.size)
        assertEquals("it's", fixes[0].replacement)
        assertEquals(FixConfidence.Low, fixes[0].confidence)
    }

    @Test
    fun tinyMarginIsIgnored() {
        val scorer = PhraseScorer(bad = mapOf("its a" to 1f, "its an" to 1f))
        assertTrue(check(GrammarChecker(scorer), "I think its a good idea ").isEmpty())
    }

    @Test
    fun correctSentenceProducesNoFixes() {
        // Penalise only the wrong variants: the sentence as typed is already the best.
        val scorer = PhraseScorer(bad = mapOf("they're car" to 6f, "there car" to 6f))
        assertTrue(check(GrammarChecker(scorer), "I like their car a lot ").isEmpty())
    }

    @Test
    fun capitalisationIsPreserved() {
        val scorer = PhraseScorer(bad = mapOf("there car" to 6f, "they're car" to 6f))
        val fixes = check(GrammarChecker(scorer), "There car is red ")
        assertEquals(1, fixes.size)
        assertEquals("Their", fixes[0].replacement)
    }

    @Test
    fun onlyCurrentSentenceIsChecked() {
        val scorer = PhraseScorer(bad = mapOf("there car" to 6f, "they're car" to 6f))
        // The mistake is in a previous, finished sentence - it must be left alone.
        val fixes = check(GrammarChecker(scorer), "I like there car. It is fast ")
        assertTrue(fixes.isEmpty())
    }

    @Test
    fun lastWordGetsExtraCaution() {
        // Margin 5: high confidence mid-sentence, but last word loses 1.5 -> only 3.5 (Low).
        val scorer = PhraseScorer(bad = mapOf("like there" to 5f, "likes there" to 5f))
        val fixes = check(GrammarChecker(scorer), "I like there ")
        assertEquals(1, fixes.size)
        assertEquals(FixConfidence.Low, fixes[0].confidence)
    }

    @Test
    fun romanUrduSentencesAreSkipped() {
        val scorer = PhraseScorer(bad = mapOf("to" to 10f, "a" to 10f))
        val checker = GrammarChecker(scorer)
        assertTrue(check(checker, "yaar mujhe nahi pata ke wo kab aa raha hai to ").isEmpty())
        assertEquals(0, scorer.calls)
    }

    @Test
    fun subjectVerbAgreement() {
        val scorer = PhraseScorer(bad = mapOf("he go" to 6f))
        val fixes = check(GrammarChecker(scorer), "Every day he go to school ")
        assertTrue(fixes.any { it.original == "go" && it.replacement == "goes" })
    }

    @Test
    fun couldOfOnlyAfterModal() {
        val scorer = PhraseScorer(bad = mapOf("could of" to 6f, "of course" to -100f))
        val fixes = check(GrammarChecker(scorer), "I could of done it ")
        assertTrue(fixes.any { it.original == "of" && it.replacement == "have" })

        // "of" not after a modal: "have" is never even considered.
        val scorer2 = PhraseScorer(bad = mapOf("cup of" to 50f))
        val fixes2 = check(GrammarChecker(scorer2), "A cup of tea please ")
        assertFalse(fixes2.any { it.original == "of" })
    }

    @Test
    fun rejectedFixesAreNotProposedAgain() {
        val scorer = PhraseScorer(bad = mapOf("there car" to 6f, "they're car" to 6f))
        val checker = GrammarChecker(scorer)
        val text = "I like there car "
        val fix = check(checker, text).single()
        val key = checker.ignoreKey(fix.original, fix.replacement, text.substring(0, fix.start))
        val again = runBlocking { checker.check(text, ignore = setOf(key)) }
        assertTrue(again.isEmpty())
    }

    @Test
    fun scorerUnavailableReturnsNoFixes() {
        val checker = GrammarChecker(ContinuationScorer { _, _ -> null })
        assertTrue(check(checker, "I like there car ").isEmpty())
    }

    @Test
    fun sentenceStartDetection() {
        val checker = GrammarChecker(ContinuationScorer { _, _ -> null })
        assertEquals(0, checker.currentSentenceStart("hello there "))
        assertEquals(6, checker.currentSentenceStart("Hello. there "))
        // A sentence that just ended is still the current one
        assertEquals(0, checker.currentSentenceStart("I like there car."))
    }
}

class ConfusionSetsTest {
    @Test
    fun alternativesAreSymmetric() {
        assertTrue(ConfusionSets.alternativesFor("their").containsAll(listOf("there", "they're")))
        assertTrue(ConfusionSets.alternativesFor("there").contains("their"))
        assertTrue(ConfusionSets.alternativesFor("It's").contains("its"))
        assertTrue(ConfusionSets.alternativesFor("banana").isEmpty())
    }

    @Test
    fun thirdPersonForms() {
        assertEquals("goes", ConfusionSets.thirdPersonForTest("go"))
        assertEquals("does", ConfusionSets.thirdPersonForTest("do"))
        assertEquals("tries", ConfusionSets.thirdPersonForTest("try"))
        assertEquals("plays", ConfusionSets.thirdPersonForTest("play"))
        assertEquals("watches", ConfusionSets.thirdPersonForTest("watch"))
        assertEquals("fixes", ConfusionSets.thirdPersonForTest("fix"))
        assertEquals("wants", ConfusionSets.thirdPersonForTest("want"))
        assertTrue(ConfusionSets.alternativesFor("goes").contains("go"))
    }

    @Test
    fun matchCase() {
        assertEquals("Their", ConfusionSets.matchCase("There", "their"))
        assertEquals("THEIR", ConfusionSets.matchCase("THERE", "their"))
        assertEquals("their", ConfusionSets.matchCase("there", "their"))
        assertEquals("An", ConfusionSets.matchCase("A", "an"))
    }
}

class RomanUrduTest {
    @Test
    fun detectsMostlyRomanUrdu() {
        assertTrue(RomanUrdu.isMostlyRomanUrdu("yaar mujhe nahi pata"))
        assertTrue(RomanUrdu.isMostlyRomanUrdu("kal mein office jaunga"))
        assertTrue(RomanUrdu.isMostlyRomanUrdu("acha theek hai, meeting ke baad call karo"))
        assertTrue(RomanUrdu.isMostlyRomanUrdu("kya scene hai aaj"))
    }

    @Test
    fun englishIsNotRomanUrdu() {
        assertFalse(RomanUrdu.isMostlyRomanUrdu("I want to go to the main office"))
        assertFalse(RomanUrdu.isMostlyRomanUrdu("Do you have the car keys"))
        assertFalse(RomanUrdu.isMostlyRomanUrdu("So what do you say about the bus"))
        assertFalse(RomanUrdu.isMostlyRomanUrdu("hi there"))
        assertFalse(RomanUrdu.isMostlyRomanUrdu(""))
    }

    @Test
    fun singleWordIsNotEnough() {
        assertFalse(RomanUrdu.isMostlyRomanUrdu("thanks yaar"))
    }

    @Test
    fun wordLookup() {
        assertTrue(RomanUrdu.isRomanUrduWord("nahi"))
        assertTrue(RomanUrdu.isRomanUrduWord("Acha"))
        assertFalse(RomanUrdu.isRomanUrduWord("hello"))
    }
}
