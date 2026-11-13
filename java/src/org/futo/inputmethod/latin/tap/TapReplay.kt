package org.futo.inputmethod.latin.tap

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Tap plan, Stage 1/5: replays your recorded taps through each key-detection rule and counts
 * how many letters come out as the letter you meant.
 *
 * Letters "meant": the word you kept (accepted, picked, or the typed word when you undid an
 * autocorrection), position by position, for words with the same length as what was typed;
 * plus mid-word fixes (the deleted tap was meant to be the replacement letter).
 */
object TapReplay {
    data class Row(val name: String, val taps: Int, val wrong: Int) {
        val wrongPer100 get() = if (taps == 0) 0f else wrong * 100f / taps
    }
    data class Report(val rows: List<Row>, val bestStrength: Float?, val note: String)

    private class Case(val keys: List<LetterKey>, val x: Float, val y: Float, val meant: Char,
                       val sentence: String, val prefix: String)

    val STRENGTHS = listOf(0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f)
    const val MAX_TAPS = 3000

    suspend fun run(context: Context, onProgress: (Float) -> Unit = {}): Report {
        val records = TapRecording.readAll(context)
        val cases = ArrayList<Case>()
        for (r in records.asReversed()) { // newest first
            if (cases.size >= MAX_TAPS) break
            val keys = LetterKey.fromBoxes(r.keys)
            if (r.kind == "fix") {
                cases.add(Case(keys, r.tapX[0].toFloat(), r.tapY[0].toFloat(), r.word[0].lowercaseChar(), "", ""))
                continue
            }
            val meant = r.word.lowercase()
            if (meant.length != r.typed.length || meant.length != r.tapX.size) continue
            if (!meant.all { it in 'a'..'z' && r.keys.index(it) >= 0 }) continue
            val sentence = r.context.joinToString(" ")
            for (i in meant.indices) {
                cases.add(Case(keys, r.tapX[i].toFloat(), r.tapY[i].toFloat(), meant[i], sentence, meant.substring(0, i)))
            }
        }
        if (cases.isEmpty()) throw IllegalStateException("No recorded taps yet. Turn on \"Record my taps\" and type for a while.")

        // Next-letter likelihoods for each tap (typing model + dictionary + your words)
        val cache = HashMap<String, FloatArray?>()
        val validCache = HashMap<String, Set<Char>?>()
        val probs = ArrayList<FloatArray?>(cases.size)
        val valids = ArrayList<Set<Char>?>(cases.size)
        var dictAvailable = false
        cases.forEachIndexed { i, c ->
            // Dictionaries are only touched on the main thread, like the keyboard does
            val valid = validCache.getOrPut(c.prefix) {
                TapPredictor.dictionaryProvider?.let { dp -> withContext(Dispatchers.Main) { dp(c.prefix) } }
                    ?.mapNotNull { cp -> Character.toLowerCase(cp).toChar().takeIf { it in 'a'..'z' } }?.toSet()
            }
            if (valid != null) dictAvailable = true
            val key = c.sentence + "|" + c.prefix
            // Same rule as the keyboard: no predictions for a word's first letter
            val p = if (c.prefix.isEmpty()) null else cache.getOrPut(key) {
                TapPredictor.computeForReplay(c.sentence, c.prefix, valid)
            }
            probs.add(p)
            valids.add(valid)
            if (i % 50 == 0) onProgress(i.toFloat() / cases.size)
        }

        val touch = TapPredictor.touchModel(context)
        fun count(rule: (Case, FloatArray?) -> Char?): Int =
            cases.indices.count { i -> rule(cases[i], probs[i]) != cases[i].meant }
        fun countIdx(rule: (Int) -> Char?): Int = cases.indices.count { i -> rule(i) != cases[i].meant }

        val rows = ArrayList<Row>()
        rows.add(Row("Nearest key (no help)", cases.size, count { c, _ -> TapDecision.underFinger(c.keys, c.x, c.y)?.letter }))
        if (dictAvailable) {
            // FUTO only boosts mid-word (not at the start of a word)
            rows.add(Row("FUTO's dictionary boost", cases.size, countIdx { i ->
                val c = cases[i]
                TapDecision.futoRule(c.keys, c.x, c.y, if (c.prefix.isEmpty()) null else valids[i])
            }))
        }
        rows.add(Row("Your touch model only", cases.size, count { c, _ ->
            TapDecision.contextRule(c.keys, c.x, c.y, null, touch, TapDecision.Params())?.letter
        }))
        var best: Pair<Float, Int>? = null
        for (s in STRENGTHS.drop(1)) {
            val wrong = count { c, p -> TapDecision.contextRule(c.keys, c.x, c.y, p, touch, TapDecision.Params(strength = s))?.letter }
            rows.add(Row("Context-aware, strength ${"%.2f".format(s)}", cases.size, wrong))
            if (best == null || wrong < best.second) best = s to wrong
        }
        onProgress(1f)
        val note = "Based on ${cases.size} of your recorded taps. Your touch model was learned from these " +
                "same taps, so its numbers are a little optimistic." +
                if (!dictAvailable) " (Open the keyboard once so FUTO's dictionary boost can be compared too.)" else ""
        return Report(rows, best?.first, note)
    }
}
