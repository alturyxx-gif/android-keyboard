package org.futo.inputmethod.latin.xlm.grammar

/**
 * Scores how likely continuations are after a prefix. In the app this is backed by the
 * keyboard's existing typing language model (LanguageModel.scoreContinuations); tests use
 * a fake.
 */
fun interface ContinuationScorer {
    /** Summed natural-log probability for each continuation, or null if unavailable. */
    suspend fun score(prefix: String, continuations: List<String>): FloatArray?
}

enum class FixConfidence { High, Low }

/**
 * A proposed fix for one word.
 *
 * [start]/[end] are character offsets into the text that was passed to [GrammarChecker.check].
 */
data class GrammarFix(
    val start: Int,
    val end: Int,
    val original: String,
    val replacement: String,
    val margin: Float,
    val confidence: FixConfidence,
)

/**
 * Live grammar checking (Phase 3, Stage 3.1).
 *
 * For every word in the current sentence that belongs to a confusion set
 * (their/there/they're, its/it's, a/an, he go/goes, ...), the keyboard's existing language
 * model scores the rest of the sentence with the original word and with each alternative.
 * Because the score includes the words *after* the candidate, the model can use the right
 * context ("there car" vs "their car").
 *
 * An alternative is only proposed when it beats the original by a clear margin:
 *  - margin >= [highThreshold]: high confidence, auto-applied
 *  - margin >= [lowThreshold]: low confidence, underlined only
 *
 * Mostly-Roman-Urdu sentences are skipped entirely.
 */
class GrammarChecker(
    private val scorer: ContinuationScorer,
    val highThreshold: Float = DEFAULT_HIGH_THRESHOLD,
    val lowThreshold: Float = DEFAULT_LOW_THRESHOLD,
) {
    companion object {
        /** Natural-log margins. 4.0 nats ~= the alternative is ~55x more likely. */
        const val DEFAULT_HIGH_THRESHOLD = 4.0f
        /** 2.0 nats ~= ~7x more likely. */
        const val DEFAULT_LOW_THRESHOLD = 2.0f

        /** The last word gets extra caution because there's no right-hand context yet. */
        const val LAST_WORD_PENALTY = 1.5f

        private val WORD_REGEX = Regex("[A-Za-z]+(?:'[A-Za-z]+)?")
        private val SENTENCE_END = Regex("[.!?\\n]")

        /** Only consider "of" -> "have" after these modals ("could of" -> "could have"). */
        private val MODALS = setOf("could", "would", "should", "must", "might", "may", "will")
    }

    data class Word(val text: String, val start: Int, val end: Int)

    /** Returns the offset where the current sentence starts within [text]. */
    fun currentSentenceStart(text: String): Int {
        var idx = -1
        for (m in SENTENCE_END.findAll(text)) {
            // Ignore a sentence terminator that is the very last character: the sentence that
            // just ended is still the one we want to check.
            if (m.range.last < text.trimEnd().length - 1) idx = m.range.last
        }
        return idx + 1
    }

    fun words(text: String, from: Int = 0): List<Word> =
        WORD_REGEX.findAll(text, from).map { Word(it.value, it.range.first, it.range.last + 1) }.toList()

    /**
     * Checks the sentence the cursor is in. [textBeforeCursor] should end with completed words
     * only (i.e. not while a word is still being composed).
     *
     * @param ignore fixes the user already rejected, as "original->replacement@context" keys
     */
    suspend fun check(textBeforeCursor: String, ignore: Set<String> = emptySet()): List<GrammarFix> {
        val sentenceStart = currentSentenceStart(textBeforeCursor)
        val sentence = textBeforeCursor.substring(sentenceStart)
        if (RomanUrdu.isMostlyRomanUrdu(sentence)) return emptyList()

        val words = words(textBeforeCursor, sentenceStart)
        if (words.size < 2) return emptyList()

        val fixes = mutableListOf<GrammarFix>()
        for ((i, w) in words.withIndex()) {
            val alternatives = ConfusionSets.alternativesFor(w.text).filter { alt ->
                // "of" -> "have" only after a modal verb
                if (w.text.equals("of", true) && alt == "have") {
                    i > 0 && words[i - 1].text.lowercase() in MODALS
                } else true
            }
            if (alternatives.isEmpty()) continue

            // Prefix: everything before this word (including earlier sentences, for context).
            // Continuation: this word + the rest of the sentence.
            val prefix = textBeforeCursor.substring(0, w.start)
            val rest = textBeforeCursor.substring(w.end).trimEnd()
            val candidates = listOf(w.text) + alternatives.map { ConfusionSets.matchCase(w.text, it) }
            val continuations = candidates.map { (it + rest) }

            val scores = scorer.score(prefix, continuations) ?: return fixes
            val original = scores[0]
            if (!original.isFinite()) continue

            var bestIdx = -1
            var bestMargin = Float.NEGATIVE_INFINITY
            for (j in 1 until scores.size) {
                if (!scores[j].isFinite()) continue
                val margin = scores[j] - original
                if (margin > bestMargin) {
                    bestMargin = margin
                    bestIdx = j
                }
            }
            if (bestIdx == -1) continue

            val isLastWord = i == words.lastIndex
            val effectiveMargin = if (isLastWord) bestMargin - LAST_WORD_PENALTY else bestMargin
            val replacement = candidates[bestIdx]
            if (ignoreKey(w.text, replacement, prefix) in ignore) continue

            val confidence = when {
                effectiveMargin >= highThreshold -> FixConfidence.High
                effectiveMargin >= lowThreshold -> FixConfidence.Low
                else -> null
            } ?: continue

            fixes.add(GrammarFix(w.start, w.end, w.text, replacement, effectiveMargin, confidence))
        }
        return fixes
    }

    /**
     * Key used to remember fixes the user undid, so they aren't proposed again for the same
     * sentence. Uses the last few words before the fix as context.
     */
    fun ignoreKey(original: String, replacement: String, prefix: String): String {
        val ctx = words(prefix).takeLast(3).joinToString(" ") { it.text.lowercase() }
        return "${original.lowercase()}->${replacement.lowercase()}@$ctx"
    }
}
