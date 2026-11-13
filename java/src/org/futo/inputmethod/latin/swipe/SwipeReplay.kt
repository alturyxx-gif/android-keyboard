package org.futo.inputmethod.latin.swipe

import android.content.Context
import org.futo.inputmethod.latin.BinaryDictionary
import org.futo.inputmethod.latin.ENCODER_ASSET
import org.futo.inputmethod.latin.ENGLISH_DECODER_ASSET
import org.futo.inputmethod.latin.ENGLISH_LM_ASSET
import org.futo.inputmethod.latin.ENGLISH_LM_VOCAB_ASSET
import org.futo.inputmethod.latin.SwipeDecoderDictionary
import org.futo.ml.inference.SwipeDecoder

/**
 * Stage 7 of the swipe plan: replays your newest recorded swipes (the 15% that fine-tuning never
 * trains on) through FUTO's model and your custom model on the phone, with the keyboard's own
 * dictionary, and reports how often each got the word right.
 */
object SwipeReplay {
    data class Result(val name: String, val n: Int, val top1: Int, val top4: Int, val avgMs: Float) {
        fun line() = "$name: first guess ${pct(top1)}, top 4 ${pct(top4)}, ${"%.1f".format(avgMs)} ms per swipe"
        private fun pct(v: Int) = if (n == 0) "-" else "${"%.1f".format(v * 100f / n)}%"
    }

    const val TEST_FRACTION = 0.15f
    const val MIN_SWIPES = 20

    /** Returns results per model, or throws IllegalStateException with a message for the user. */
    fun run(context: Context, onProgress: (Float) -> Unit = {}): List<Result> {
        val all = SwipeRecording.readAll(context).sortedBy { it.time }
        val nTest = (all.size * TEST_FRACTION).toInt()
        if (nTest < MIN_SWIPES) {
            throw IllegalStateException("Needs at least ${(MIN_SWIPES / TEST_FRACTION).toInt()} recorded swipes (you have ${all.size}).")
        }
        val test = all.takeLast(nTest).filter { it.frame != null && it.path.x.size >= 3 }
        val tries = SwipeDecoderDictionary.appliedTries
        val weights = SwipeDecoderDictionary.appliedTrieWeights
        if (tries == null || tries.isEmpty()) {
            throw IllegalStateException("Open the keyboard and swipe once first (the dictionary isn't loaded yet).")
        }

        val futo = SwipeDecoderDictionary.getFilePath(context, ENCODER_ASSET)
        val models = mutableListOf("FUTO" to futo)
        if (CustomSwipeModel.isInstalled(context)) {
            models.add(("Custom (" + (CustomSwipeModel.info(context)?.displayName ?: "custom") + ")") to
                    CustomSwipeModel.encoderFile(context).absolutePath)
        }
        val refinerPath = SwipeDecoderDictionary.getFilePath(context, ENGLISH_DECODER_ASSET)
        val lmPath = SwipeDecoderDictionary.getFilePath(context, ENGLISH_LM_ASSET)
        val lmVocab = SwipeDecoderDictionary.getFilePath(context, ENGLISH_LM_VOCAB_ASSET)

        val total = models.size * test.size
        var done = 0
        return models.map { (name, path) ->
            val isFuto = path == futo
            var top1 = 0
            var top4 = 0
            var nanos = 0L
            SwipeDecoder(encoderPath = path, beamWidth = 300, useExpansion = false).use { d ->
                var currentFrame: SwipeModelFrame? = null
                for (r in test) {
                    val f = r.frame!!
                    if (f != currentFrame) {
                        synchronized(BinaryDictionary.sTrieUsageLock) {
                            d.setMode(
                                letters = f.letters, cx = f.xs.toFloatArray(), cy = f.ys.toFloatArray(),
                                tries = tries,
                                decoderPath = if (isFuto && f.refiner) refinerPath else "",
                                lmModelPath = if (f.contextModel) lmPath else "",
                                lmVocabPath = if (f.contextModel) lmVocab else "",
                            )
                        }
                        currentFrame = f
                    }
                    val seg = SwipeDecoder.SwipeSeg(
                        x = r.path.x.map { it * f.sx + f.ox }.toFloatArray(),
                        y = r.path.y.map { minOf(1.0f, it * (4.0f / 3.0f) * f.sy + f.oy) }.toFloatArray(),
                        t = r.path.t.toFloatArray(),
                    )
                    d.setContext(r.context)
                    val t0 = System.nanoTime()
                    val results = synchronized(BinaryDictionary.sTrieUsageLock) {
                        d.recognize(arrayOf(seg), emptyArray(), topK = 4, beamWidth = 300, trieWeights = weights)
                    }
                    nanos += System.nanoTime() - t0
                    val target = norm(r.word)
                    val words = results.map { norm(it.word) }
                    if (words.firstOrNull() == target) top1++
                    if (target in words.take(4)) top4++
                    done++
                    if (done % 10 == 0) onProgress(done.toFloat() / total)
                }
            }
            Result(name, test.size, top1, top4, nanos / 1e6f / test.size.coerceAtLeast(1))
        }
    }

    private fun norm(w: String) = w.lowercase().filter { it.isLetter() }
}
