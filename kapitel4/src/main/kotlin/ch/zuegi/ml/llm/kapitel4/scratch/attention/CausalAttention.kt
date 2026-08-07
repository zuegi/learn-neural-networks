package ch.zuegi.ml.llm.kapitel4.scratch.attention

import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Single-Head Causal Self-Attention (nur Forward-Pass) mit optionalem Attention-Dropout.
 *
 * Position i darf nur auf Positionen j <= i schauen.
 *
 * Ablauf:
 *   Q = X * Wq
 *   K = X * Wk
 *   V = X * Wv
 *   scores  = Q * K^T / sqrt(dK)
 *   scores  = causalMask(scores)
 *   weights = softmax(scores)
 *   weights = dropout(weights)      // nur wenn training = true und dropoutProb > 0
 *   output  = weights * V
 */
class CausalAttention(
    private val embeddingDim: Int,
    private val dK: Int,
    private val dropoutProb: Double = 0.0,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
        require(dropoutProb in 0.0..1.0) { "dropoutProb muss in [0.0, 1.0] liegen" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val wQuery: Array<DoubleArray> = randomMatrix(embeddingDim, dK)
    val wKey: Array<DoubleArray> = randomMatrix(embeddingDim, dK)
    val wValue: Array<DoubleArray> = randomMatrix(embeddingDim, dK)

    /**
     * @param input Matrix [contextLength, embeddingDim]
     * @param training true = Dropout aktiv, false = kein Dropout
     */
    fun forward(
        input: Array<DoubleArray>,
        training: Boolean = false,
    ): Array<DoubleArray> {
        require(input.isNotEmpty()) { "input darf nicht leer sein" }
        require(input[0].size == embeddingDim) {
            "Input-Embedding-Dim ${input[0].size} passt nicht zu embeddingDim $embeddingDim"
        }

        val query = matMul(input, wQuery)
        val key = matMul(input, wKey)
        val value = matMul(input, wValue)

        val scores = scaledScores(query, key)
        maskCausal(scores)

        var weights = softmaxRows(scores)
        if (training && dropoutProb > 0.0) {
            weights = applyDropout(weights, dropoutProb)
        }

        return matMul(weights, value)
    }

    private fun scaledScores(
        query: Array<DoubleArray>,
        key: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val scale = sqrt(dK.toDouble())
        val ctx = query.size

        return Array(ctx) { i ->
            DoubleArray(ctx) { j ->
                var sum = 0.0
                for (d in 0 until dK) {
                    sum += query[i][d] * key[j][d]
                }
                sum / scale
            }
        }
    }

    private fun maskCausal(scores: Array<DoubleArray>) {
        for (i in scores.indices) {
            for (j in i + 1 until scores[i].size) {
                scores[i][j] = Double.NEGATIVE_INFINITY
            }
        }
    }

    private fun softmaxRows(scores: Array<DoubleArray>): Array<DoubleArray> =
        Array(scores.size) { i ->
            val row = scores[i]
            val max = row.max()
            val exps = DoubleArray(row.size) { exp(row[it] - max) }
            val sum = exps.sum()
            DoubleArray(row.size) { exps[it] / sum }
        }

    /**
     * Inverted Dropout:
     * - mit Wahrscheinlichkeit p -> 0
     * - sonst durch (1 - p) teilen, damit Erwartungswert gleich bleibt
     */
    private fun applyDropout(
        weights: Array<DoubleArray>,
        p: Double,
    ): Array<DoubleArray> {
        if (p == 0.0) return weights
        if (p == 1.0) {
            return Array(weights.size) { DoubleArray(weights[it].size) { 0.0 } }
        }

        val keepScale = 1.0 / (1.0 - p)
        return Array(weights.size) { i ->
            DoubleArray(weights[i].size) { j ->
                if (rnd.nextDouble() < p) 0.0 else weights[i][j] * keepScale
            }
        }
    }

    private fun matMul(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val rows = a.size
        val inner = b.size
        val cols = b[0].size

        return Array(rows) { i ->
            DoubleArray(cols) { j ->
                var sum = 0.0
                for (k in 0 until inner) {
                    sum += a[i][k] * b[k][j]
                }
                sum
            }
        }
    }

    private fun randomMatrix(
        rows: Int,
        cols: Int,
    ): Array<DoubleArray> = Array(rows) { DoubleArray(cols) { rnd.nextGaussian() * INIT_SCALE } }

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
