package org.futo.inputmethod.latin.tap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/** QWERTY with 100x150 px keys (like a phone keyboard in key-detection pixels). */
private fun qwerty(): KeyBoxes {
    val rows = listOf("qwertyuiop", "asdfghjkl", "zxcvbnm")
    val offs = listOf(0, 50, 150)
    val letters = StringBuilder(); val x = ArrayList<Int>(); val y = ArrayList<Int>()
    rows.forEachIndexed { r, row -> row.forEachIndexed { i, c -> letters.append(c); x.add(offs[r] + i * 100); y.add(r * 150) } }
    return KeyBoxes(1000, 450, letters.toString(), x, y, List(x.size) { 100 }, List(x.size) { 150 })
}
private val KB = qwerty()
private val KEYS = LetterKey.fromBoxes(KB)
private fun centre(c: Char) = KB.centre(KB.index(c))
private fun probs(vararg pairs: Pair<Char, Float>): FloatArray {
    val p = FloatArray(26) { 0.01f }
    pairs.forEach { (c, v) -> p[c - 'a'] = v }
    val s = p.sum(); for (i in p.indices) p[i] /= s
    return p
}

class TapDecisionTest {
    private val params = TapDecision.Params(strength = 1.0f)

    @Test fun centreTapNeverChangesEvenWithStrongPrediction() {
        val (cx, cy) = centre('e')
        val r = TapDecision.contextRule(KEYS, cx + 10, cy + 10, probs('r' to 0.95f), null, params)!!
        assertEquals('e', r.letter); assertFalse(r.changed)
    }

    @Test fun edgeTapGoesToLikelyNeighbour() {
        val (cx, cy) = centre('e')
        // 45 px right of e's centre: on e, near the e|r edge
        val r = TapDecision.contextRule(KEYS, cx + 45, cy, probs('r' to 0.9f), null, params)!!
        assertEquals('r', r.letter); assertTrue(r.changed); assertEquals('e', r.underFinger)
    }

    @Test fun flatPredictionLeavesEdgeTapAlone() {
        val (cx, cy) = centre('e')
        val flat = FloatArray(26) { 1f / 26 }
        assertEquals('e', TapDecision.contextRule(KEYS, cx + 45, cy, flat, null, params)!!.letter)
    }

    @Test fun neverJumpsToAFarKey() {
        val (cx, cy) = centre('q')
        // Even if 'p' is almost certain, a tap at q's edge can't become p
        val r = TapDecision.contextRule(KEYS, cx + 45, cy, probs('p' to 0.99f), null, params)!!
        assertTrue(r.letter == 'q' || r.letter == 'w')
    }

    @Test fun tapsInTheOuterMarginAreNotMoved() {
        val (lx, ly) = centre('l')
        // Well to the right of l (outside every letter key): never moved to p, even if p is likely
        val r = TapDecision.contextRule(KEYS, lx + 120, ly, probs('p' to 0.95f), null, params)!!
        assertEquals('l', r.letter); assertFalse(r.changed)
    }

    @Test fun tapsInTheGapBetweenRowsCanStillBeMoved() {
        val (ex, ey) = centre('e')
        // Just below e, in a small gap (inside 0.15 key): allowed to go to the likely letter
        val keys = KEYS.map { if (it.letter in "asdfghjkl") it.copy(top = it.top + 10) else it }
        val r = TapDecision.contextRule(keys, ex + 40, ey + 80, probs('r' to 0.95f), null, params)!!
        assertEquals('r', r.letter)
    }

    @Test fun strengthZeroIsTouchOnly() {
        val (cx, cy) = centre('e')
        val r = TapDecision.contextRule(KEYS, cx + 45, cy, probs('r' to 0.9f), null, TapDecision.Params(strength = 0f))!!
        assertEquals('e', r.letter)
    }

    @Test fun futoRuleBoostsValidLettersNearEdge() {
        val (cx, cy) = centre('r')
        // On t, 20 px from the r|t edge: FUTO's boost gives r if only r is valid
        assertEquals('r', TapDecision.futoRule(KEYS, cx + 70, cy, setOf('r')))
        assertEquals('t', TapDecision.futoRule(KEYS, cx + 70, cy, null))
    }
}

class TouchModelTest {
    @Test fun learnsConsistentOffsetAfterEnoughTaps() {
        val tm = TouchModel()
        repeat(TouchModel.MIN_TAPS - 1) { tm.learn('n', 0.05f, 0.3f) }
        assertEquals(0f, tm.forKey('n')[1], 1e-6f) // not used yet
        repeat(200) { tm.learn('n', 0.05f, 0.3f) }
        assertEquals(0.3f, tm.forKey('n')[1], 0.02f)
    }

    @Test fun ignoresOutliers() {
        val tm = TouchModel()
        repeat(100) { tm.learn('a', 0f, 0f) }
        repeat(10) { tm.learn('a', 2f, 2f) }
        assertEquals(0f, tm.forKey('a')[0], 1e-6f)
    }

    @Test fun learnedOffsetMovesTheKey() {
        val tm = TouchModel()
        repeat(300) { tm.learn('v', 0f, 0.35f) } // always hits v low
        val (cx, cy) = centre('v')
        // A tap low on v, just inside the space below, still counts as v with the touch model
        val r = TapDecision.contextRule(KEYS, cx, cy + 60, null, tm, TapDecision.Params())!!
        assertEquals('v', r.letter)
    }
}

class LetterMixTest {
    @Test fun dictionaryDownweightsInvalidLetters() {
        val lm = FloatArray(26) { 1f / 26 }
        val p = LetterMix.combine(lm, setOf('r'), emptyList(), "sto")!!
        assertTrue(p['r' - 'a'] > 3 * p['x' - 'a'])
        assertEquals(1f, p.sum(), 1e-4f)
    }

    @Test fun personalWordsGetASay() {
        val lm = FloatArray(26) { 1f / 26 }
        val p = LetterMix.combine(lm, null, listOf("zarnaab" to 10), "zar")!!
        assertTrue(p['n' - 'a'] > 0.2f)
    }

    @Test fun nothingKnownGivesNull() = assertNull(LetterMix.combine(null, null, emptyList(), ""))

    @Test fun dictionaryOnlyIsUniformOverValid() {
        val p = LetterMix.combine(null, setOf('a', 'e'), emptyList(), "th")!!
        assertEquals(0.5f, p['a' - 'a'], 1e-5f)
        assertEquals(0f, p['z' - 'a'], 1e-6f)
    }
}

private fun commit(typed: String, committed: String, manual: Boolean = false): TapCommit {
    val xs = typed.map { centre(it).first.toInt() }.toIntArray()
    val ys = typed.map { centre(it).second.toInt() }.toIntArray()
    return TapCommit(1L, typed, committed, manual, xs, ys, listOf("say"), KB)
}

class TapLabelerTest {
    @Test fun wordIsRecordedWhenTheNextOneStarts() {
        val l = TapLabeler()
        assertTrue(l.onCommit(commit("hello", "hello")).isEmpty())
        val r = l.onCommit(commit("there", "there")).single()
        assertEquals("hello", r.word); assertEquals(TapSource.ACCEPTED, r.source)
    }

    @Test fun autocorrectedAndPicked() {
        val l = TapLabeler()
        l.onCommit(commit("tge", "the"))
        assertEquals(TapSource.AUTOCORRECTED, l.onCommit(commit("cat", "car", manual = true)).single().source)
        assertEquals(TapSource.PICKED, l.onFinishInput().single().source)
    }

    @Test fun undoingAutocorrectMeansTypedWasRight() {
        val l = TapLabeler()
        l.onCommit(commit("zarnaab", "zebra"))
        l.onEditCommitted() // the backspace...
        val r = l.onRevert().single() // ...was an autocorrect undo
        assertEquals("zarnaab", r.word); assertEquals(TapSource.REVERTED, r.source)
    }

    @Test fun editingACommittedWordDropsIt() {
        val l = TapLabeler()
        l.onCommit(commit("hello", "hello"))
        l.onEditCommitted()
        assertTrue(l.onCommit(commit("there", "there")).isEmpty())
    }

    @Test fun midWordFixRecordsTheWrongTap() {
        val l = TapLabeler()
        val (ex, ey) = centre('e'); val (rx, ry) = centre('r')
        l.onLetterDeleted('e', (ex + 45).toInt(), ey.toInt(), 2)
        val fix = l.onLetterTyped('r', rx.toInt(), ry.toInt(), 2, 5L, emptyList(), KB)!!
        assertEquals("e", fix.typed); assertEquals("r", fix.word); assertEquals(TapSource.FIX, fix.source)
    }

    @Test fun retypingTheSameLetterOrFarAwayIsNotAFix() {
        val l = TapLabeler()
        val (ex, ey) = centre('e')
        l.onLetterDeleted('e', ex.toInt(), ey.toInt(), 2)
        assertNull(l.onLetterTyped('e', ex.toInt(), ey.toInt(), 2, 5L, emptyList(), KB))
        val (mx, my) = centre('m')
        l.onLetterDeleted('e', ex.toInt(), ey.toInt(), 2)
        assertNull(l.onLetterTyped('m', mx.toInt(), my.toInt(), 2, 5L, emptyList(), KB))
    }
}

class TapKeyFeedbackTest {
    @Test fun backsOffWhenChangedTapsKeepGettingDeleted() {
        TapKeyFeedback.reset()
        assertEquals(1f, TapKeyFeedback.strengthMultiplier(), 1e-6f)
        val m = TapKeyFeedback::class.java.getDeclaredMethod("record", Boolean::class.java).apply { isAccessible = true }
        repeat(30) { m.invoke(TapKeyFeedback, true) }
        assertTrue(TapKeyFeedback.strengthMultiplier() < 0.5f)
        TapKeyFeedback.reset()
        repeat(30) { m.invoke(TapKeyFeedback, false) }
        assertEquals(1f, TapKeyFeedback.strengthMultiplier(), 1e-6f)
        assertTrue(abs(TapKeyFeedback.strengthMultiplier() - 1f) < 1e-6f)
    }
}
