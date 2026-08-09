package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import java.util.Random
import kotlin.math.PI
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * Forward-only position-weises Feed-Forward-Modul mit multik.
 *
 * Verarbeitet eine Token-Zeile durch zwei lineare Schichten mit GELU:
 *
 *     h   = gelu(W1 · x + b1)   // W1: [hiddenDim, embeddingDim]
 *     out = W2 · h + b2         // W2: [embeddingDim, hiddenDim]
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param hiddenDim versteckte Dimension, Standard 4 * embeddingDim.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class FeedForwardMultik(
    private val embeddingDim: Int,
    private val hiddenDim: Int = 4 * embeddingDim,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    // W1: [hiddenDim, embeddingDim]
    val w1: NDArray<Double, D2> =
        mk.ndarray(
            List(hiddenDim) {
                List(embeddingDim) { rnd.nextGaussian() * INIT_SCALE }
            },
        )

    // b1: [hiddenDim]
    val b1: NDArray<Double, D1> = mk.ndarray(DoubleArray(hiddenDim))

    // W2: [embeddingDim, hiddenDim]
    val w2: NDArray<Double, D2> =
        mk.ndarray(
            List(embeddingDim) {
                List(hiddenDim) { rnd.nextGaussian() * INIT_SCALE }
            },
        )

    // b2: [embeddingDim]
    val b2: NDArray<Double, D1> = mk.ndarray(DoubleArray(embeddingDim))

    /**
     * Forward-Pass fuer eine Token-Zeile.
     *
     * @param x Vektor [embeddingDim].
     * @return Vektor [embeddingDim].
     */
    fun forward(x: NDArray<Double, D1>): NDArray<Double, D1> {
        require(x.size == embeddingDim) {
            "x.size ${x.size} passt nicht zu embeddingDim $embeddingDim"
        }

        val hiddenPre = matVecMul(w1, x, rows = hiddenDim, cols = embeddingDim)
        val hidden = mk.ndarray(DoubleArray(hiddenDim) { gelu(hiddenPre[it] + b1[it]) })

        val outPre = matVecMul(w2, hidden, rows = embeddingDim, cols = hiddenDim)
        return mk.ndarray(DoubleArray(embeddingDim) { outPre[it] + b2[it] })
    }

    private fun matVecMul(
        weight: NDArray<Double, D2>,
        vector: NDArray<Double, D1>,
        rows: Int,
        cols: Int,
    ): NDArray<Double, D1> {
        val result = DoubleArray(rows)
        for (r in 0 until rows) {
            var sum = 0.0
            for (c in 0 until cols) {
                sum += weight[r][c] * vector[c]
            }
            result[r] = sum
        }
        return mk.ndarray(result)
    }

    private fun gelu(x: Double): Double {
        val c = sqrt(2.0 / PI)
        val inner = c * (x + GELU_COEFF * x * x * x)
        return 0.5 * x * (1.0 + tanh(inner))
    }

    companion object {
        private const val INIT_SCALE = 0.02
        private const val GELU_COEFF = 0.044715
    }
}
