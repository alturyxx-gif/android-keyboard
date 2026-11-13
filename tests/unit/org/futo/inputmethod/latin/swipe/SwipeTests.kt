package org.futo.inputmethod.latin.swipe

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

private const val LETTERS = "abcdefghijklmnopqrstuvwxyz"
// The English refiner's expected positions (copied from SwipeDecoderDictionary.SpecialDecoder)
private val REF_X = listOf(0.055555555555555566f, 0.611111111111111f, 0.38888888888888895f, 0.2777777777777778f, 0.22222222222222227f, 0.38888888888888895f, 0.5000000000000001f, 0.611111111111111f, 0.7777777777777778f, 0.7222222222222222f, 0.8333333333333334f, 0.9444444444444445f, 0.8333333333333334f, 0.7222222222222222f, 0.888888888888889f, 1.0f, 0.0f, 0.33333333333333337f, 0.1666666666666667f, 0.44444444444444453f, 0.6666666666666667f, 0.5000000000000001f, 0.11111111111111112f, 0.2777777777777778f, 0.5555555555555556f, 0.1666666666666667f)
private val REF_Y = listOf(0.5f, 1.0f, 1.0f, 0.5f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f)
private const val MAX_DX = 0.09f
private const val MAX_DY = 0.06666f

/** Key centres of a QWERTY keyboard, in pixels, with the given size and padding. */
private fun qwerty(width: Float, rowHeight: Float, topPad: Float = 0f, row3Shift: Float = 0f): Map<Char, Pair<Float, Float>> {
    val rows = listOf("qwertyuiop", "asdfghjkl", "zxcvbnm")
    val keyW = width / 10f
    val starts = listOf(0f, 0.5f * keyW, 1.5f * keyW + row3Shift)
    val out = HashMap<Char, Pair<Float, Float>>()
    rows.forEachIndexed { r, row ->
        row.forEachIndexed { i, c ->
            out[c] = (starts[r] + keyW * (i + 0.5f)) to (topPad + rowHeight * (r + 0.5f))
        }
    }
    return out
}

private fun match(keys: Map<Char, Pair<Float, Float>>) =
    SwipeLayoutMatch.compare(LETTERS, REF_X, REF_Y, MAX_DX, MAX_DY, LETTERS.map { keys[it]!! })

class SwipeLayoutMatchTest {
    @Test fun normalKeyboardMatches() = assertTrue(match(qwerty(1440f, 160f)).matches)

    @Test fun loweredKeyboardMatches() {
        // Shorter keys (like a reduced keyboard height) and extra padding
        val r = match(qwerty(1440f, 100f, topPad = 30f))
        assertTrue(r.reason, r.matches)
    }

    @Test fun raisedKeyboardMatches() = assertTrue(match(qwerty(1440f, 260f)).matches)

    @Test fun narrowPhoneMatches() = assertTrue(match(qwerty(720f, 120f)).matches)

    @Test fun shiftedBottomRowDoesNotMatchAndSaysWhy() {
        val r = match(qwerty(1440f, 160f, row3Shift = 144f)) // bottom row one key to the right
        assertFalse(r.matches)
        assertTrue(r.reason, r.reason.contains("too far"))
        assertTrue(r.worstKey in "zxcvbnm".toSet() || r.worstKey in "qp".toSet())
    }
}

private fun capture(
    candidates: List<String>,
    start: Char = 'h', end: Char = 'o',
    time: Long = 1000L,
): SwipeCapture {
    val keys = qwerty(1f, 1f / 3f)
    val k = SwipeKeys(LETTERS, LETTERS.map { keys[it]!!.first }, LETTERS.map { keys[it]!!.second })
    val s = keys[start]!!; val e = keys[end]!!
    return SwipeCapture(time, SwipePath(listOf(s.first, e.first), listOf(s.second, e.second), listOf(0f, 300f)),
        k, null, candidates, listOf("say"))
}

class SwipeLabelerTest {
    private var now = 1000L
    private val labeler = SwipeLabeler { now }

    @Test fun acceptedWord() {
        labeler.onSwipe(capture(listOf("hello", "jello")))
        val r = labeler.onCommitted("Hello")
        assertEquals(1, r.size)
        assertEquals("Hello", r[0].word)
        assertEquals(SwipeSource.ACCEPTED, r[0].source)
    }

    @Test fun pickedAlternative() {
        labeler.onSwipe(capture(listOf("jello", "hello")))
        val r = labeler.onCommitted("hello")
        assertEquals(SwipeSource.PICKED, r.single().source)
    }

    @Test fun unrelatedCommitIsDropped() {
        labeler.onSwipe(capture(listOf("jello", "hello")))
        assertTrue(labeler.onCommitted("banana").isEmpty())
    }

    @Test fun deletedAndRetypedIsLabelledWithRetypedWord() {
        labeler.onSwipe(capture(listOf("hippo", "jello")))
        labeler.onRejected()
        now += 5000
        val r = labeler.onCommitted("hello")
        assertEquals("hello", r.single().word)
        assertEquals(SwipeSource.RETYPED, r.single().source)
    }

    @Test fun deletedThenSwipedAgainLabelsBoth() {
        labeler.onSwipe(capture(listOf("hippo")))
        labeler.onRejected()
        now += 2000
        labeler.onSwipe(capture(listOf("hello"), time = now))
        val r = labeler.onCommitted("hello")
        assertEquals(setOf(SwipeSource.RETYPED, SwipeSource.ACCEPTED), r.map { it.source }.toSet())
    }

    @Test fun retypedWordMustFitThePath() {
        labeler.onSwipe(capture(listOf("hippo"), start = 'h', end = 'o'))
        labeler.onRejected()
        // Changed their mind: "zebra" starts and ends far from the swipe
        assertTrue(labeler.onCommitted("zebra").isEmpty())
    }

    @Test fun retypeAfterTooLongIsDropped() {
        labeler.onSwipe(capture(listOf("hippo")))
        labeler.onRejected()
        now += SwipeLabeler.RETYPE_WINDOW_MS + 1
        assertTrue(labeler.onCommitted("hello").isEmpty())
    }

    @Test fun onlyTheNextWordCountsAsRetype() {
        labeler.onSwipe(capture(listOf("hippo")))
        labeler.onRejected()
        labeler.onCommitted("ho") // something else first
        assertTrue(labeler.onCommitted("hello").isEmpty())
    }

    @Test fun leavingTheFieldDropsPending() {
        labeler.onSwipe(capture(listOf("hello")))
        labeler.onFinishInput()
        assertTrue(labeler.onCommitted("hello").isEmpty())
    }

    @Test fun recordRoundTripsThroughJson() {
        labeler.onSwipe(capture(listOf("hello", "jello")))
        val rec = labeler.onCommitted("hello").single()
        val json = Json { encodeDefaults = true }
        val text = json.encodeToString(SwipeRecord.serializer(), rec)
        assertEquals(rec, json.decodeFromString(SwipeRecord.serializer(), text))
        assertTrue(text.contains("\"word\":\"hello\""))
    }
}
