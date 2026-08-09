package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import java.util.Random

/**
 * Trainierbares Feed-Forward-Modul auf Basis von [TensorMultik].
 *
 * Position-weise zwei lineare Schichten mit GELU:
 *     h   = gelu(W1 · x + b1)   // W1: [hiddenDim, embeddingDim]
 *     out = W2 · h + b2         // W2: [embeddingDim, hiddenDim]
 */
class FeedForwardMultikTensor(
    private val embeddingDim: Int,
    private val hiddenDim: Int = 4 * embeddingDim,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    // W1: [hiddenDim, embeddingDim], flach row-major
    val w1: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(hiddenDim * embeddingDim) { rnd.nextGaussian() * INIT_SCALE }),
        )
    val b1: TensorMultik = TensorMultik(mk.ndarray(DoubleArray(hiddenDim)))

    // W2: [embeddingDim, hiddenDim], flach row-major
    val w2: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(embeddingDim * hiddenDim) { rnd.nextGaussian() * INIT_SCALE }),
        )
    val b2: TensorMultik = TensorMultik(mk.ndarray(DoubleArray(embeddingDim)))

    /**
     * Forward-Pass fuer eine Token-Zeile.
     *
     * @param x Eingabe-Tensor der Laenge [embeddingDim].
     * @return Output-Tensor der Laenge [embeddingDim].
     */
    fun forward(x: TensorMultik): TensorMultik {
        val hidden = (x.matVecMul(w1, m = hiddenDim, n = embeddingDim) + b1).gelu()
        return hidden.matVecMul(w2, m = embeddingDim, n = hiddenDim) + b2
    }

    fun parameters(): List<TensorMultik> = listOf(w1, b1, w2, b2)

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
