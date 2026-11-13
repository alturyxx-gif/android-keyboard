package org.futo.inputmethod.latin.tap

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Handler
import android.os.Looper
import android.view.View
import org.futo.inputmethod.keyboard.Keyboard
import java.lang.ref.WeakReference

/**
 * Debug view (Tap plan, Stage 4.4): shades the areas where a tap would now go to a different key
 * than the one under the finger (coloured per key it goes to), and puts a dot where your taps on
 * each key usually land (your touch model). Only drawn when "Show key areas" is on.
 */
object TapOverlay {
    private const val STEP = 10 // px grid
    private var view: WeakReference<View>? = null
    private val main = Handler(Looper.getMainLooper())
    private var lastVersion = -1
    private var cells: IntArray = IntArray(0) // x, y, letter index per changed cell
    private var dots: FloatArray = FloatArray(0)
    private val fill = Paint().apply { style = Paint.Style.FILL }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(220, 255, 64, 64); style = Paint.Style.FILL }
    private var watching = false

    /** Called by the keyboard view; it's redrawn whenever the predictions change. */
    @JvmStatic
    fun attach(v: View) {
        view = WeakReference(v)
        if (!watching) {
            watching = true
            main.post(object : Runnable {
                var seen = -1
                override fun run() {
                    val vv = view?.get()
                    if (vv != null && TapPredictor.debugOverlay && TapPredictor.overlayVersion != seen) {
                        seen = TapPredictor.overlayVersion
                        vv.invalidate()
                    }
                    main.postDelayed(this, if (TapPredictor.debugOverlay) 50L else 500L)
                }
            })
        }
    }

    /** [offX]/[offY]: add to key coordinates to get view coordinates. */
    @JvmStatic
    fun draw(canvas: Canvas, context: Context, keyboard: Keyboard?, offX: Int, offY: Int) {
        if (!TapPredictor.debugOverlay || keyboard == null) return
        if (TapPredictor.overlayVersion != lastVersion) {
            lastVersion = TapPredictor.overlayVersion
            compute(context, keyboard)
        }
        var i = 0
        while (i < cells.size) {
            fill.color = colorFor(cells[i + 2])
            canvas.drawRect((cells[i] + offX).toFloat(), (cells[i + 1] + offY).toFloat(),
                (cells[i] + offX + STEP).toFloat(), (cells[i + 1] + offY + STEP).toFloat(), fill)
            i += 3
        }
        i = 0
        while (i < dots.size) {
            canvas.drawCircle(dots[i] + offX, dots[i + 1] + offY, 6f, dotPaint)
            i += 2
        }
    }

    private fun compute(context: Context, keyboard: Keyboard) {
        val keys = TapKeyResolver.lettersOf(keyboard)
        if (keys.isEmpty()) { cells = IntArray(0); dots = FloatArray(0); return }
        val touch = TapPredictor.touchModel(context)
        val snap = TapPredictor.current()
        val params = TapDecision.Params(strength = TapPredictor.strength * TapKeyFeedback.strengthMultiplier())
        val minX = keys.minOf { it.left }.toInt()
        val maxX = keys.maxOf { it.left + it.w }.toInt()
        val minY = keys.minOf { it.top }.toInt()
        val maxY = keys.maxOf { it.top + it.h }.toInt()
        val out = ArrayList<Int>()
        if (TapPredictor.enabled) {
            var y = minY
            while (y < maxY) {
                var x = minX
                while (x < maxX) {
                    val r = TapDecision.contextRule(keys, x + STEP / 2f, y + STEP / 2f, snap?.probs, touch, params)
                    if (r != null && r.changed) {
                        out.add(x); out.add(y); out.add(r.letter - 'a')
                    }
                    x += STEP
                }
                y += STEP
            }
        }
        cells = out.toIntArray()
        val d = ArrayList<Float>()
        for (k in keys) {
            val o = touch.forKey(k.letter)
            if (o[0] != 0f || o[1] != 0f) { d.add(k.cx + o[0] * k.w); d.add(k.cy + o[1] * k.h) }
        }
        dots = d.toFloatArray()
    }

    private fun colorFor(i: Int): Int {
        val hsv = floatArrayOf((i * 137.5f) % 360f, 0.8f, 1f)
        return Color.HSVToColor(90, hsv)
    }
}
