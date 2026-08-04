package ch.zuegi.ml.llm

import java.util.Random
import kotlin.math.PI
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * Position-weises Feed-Forward-Netz (nur Forward-Pass).
 *
 * Verarbeitet jedes Token unabhaengig durch zwei lineare Schichten mit
 * GELU-Aktivierung dazwischen. Die versteckte Dimension ist ueblicherweise
 * ein Vielfaches von embeddingDim (Standard: 4x).
 *
 * Pro Token-Zeile:
 *
 *     h   = gelu(x * W1 + b1)   // W1: [embeddingDim, hiddenDim]
 *     out = h * W2 + b2         // W2: [hiddenDim, embeddingDim]
 *
 * Input und Output haben dieselbe Form [contextLength, embeddingDim],
 * damit Residual-Verbindungen moeglich sind.
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param hiddenDim versteckte Dimension, Standard 4 * embeddingDim.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class FeedForward(
    private val embeddingDim: Int,
    private val hiddenDim: Int = 4 * embeddingDim,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val w1: Array<DoubleArray> = randomMatrix(embeddingDim, hiddenDim)
    val b1: DoubleArray = DoubleArray(hiddenDim)

    val w2: Array<DoubleArray> = randomMatrix(hiddenDim, embeddingDim)
    val b2: DoubleArray = DoubleArray(embeddingDim)

    /**
     * Forward-Pass fuer alle Token-Zeilen.
     *
     * @param input Form [contextLength, embeddingDim].
     * @return Output gleicher Form [contextLength, embeddingDim].
     */
    fun forward(input: Array<DoubleArray>): Array<DoubleArray> = Array(input.size) { i -> forwardRow(input[i]) }

    private fun forwardRow(x: DoubleArray): DoubleArray {
        val hidden = linear(x, w1, b1)
        val activated = DoubleArray(hidden.size) { gelu(hidden[it]) }
        return linear(activated, w2, b2)
    }

    private fun linear(
        x: DoubleArray,
        weight: Array<DoubleArray>,
        bias: DoubleArray,
    ): DoubleArray {
        val cols = weight[0].size
        return DoubleArray(cols) { j ->
            var sum = bias[j]
            for (k in x.indices) {
                sum += x[k] * weight[k][j]
            }
            sum
        }
    }

    private fun gelu(x: Double): Double {
        val c = sqrt(2.0 / PI)
        val inner = c * (x + GELU_COEFF * x.pow(3))
        return 0.5 * x * (1.0 + tanh(inner))
    }

    private fun randomMatrix(
        rows: Int,
        cols: Int,
    ): Array<DoubleArray> = Array(rows) { DoubleArray(cols) { rnd.nextGaussian() * INIT_SCALE } }

    companion object {
        private const val INIT_SCALE = 0.02
        private const val GELU_COEFF = 0.044715
    }
}
