package ch.zuegi.ml.llm

import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Einfache Single-Head Self-Attention (nur Forward-Pass).
 *
 * Optional mit Causal Masking: Position i darf nur auf Positionen j <= i schauen.
 *
 * Ablauf pro Sequenz:
 *
 *     Q = X * Wq
 *     K = X * Wk
 *     V = X * Wv
 *     scores  = Q * K^T / sqrt(dK)
 *     scores  = maskCausal(scores)   // nur wenn causal = true
 *     weights = softmax(scores)
 *     output  = weights * V
 *
 * @param embeddingDim Laenge der Input-Embeddings.
 * @param dK Dimension von Query/Key/Value.
 * @param causal wenn true, wird Zukunft maskiert.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class SelfAttention(
    private val embeddingDim: Int,
    private val dK: Int,
    private val causal: Boolean = false,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val wQuery: Array<DoubleArray> = randomMatrix(embeddingDim, dK)
    val wKey: Array<DoubleArray> = randomMatrix(embeddingDim, dK)
    val wValue: Array<DoubleArray> = randomMatrix(embeddingDim, dK)

    fun forward(input: Array<DoubleArray>): Array<DoubleArray> {
        val query = matMul(input, wQuery)
        val key = matMul(input, wKey)
        val value = matMul(input, wValue)

        val scores = scaledScores(query, key)
        if (causal) {
            maskCausal(scores)
        }
        val weights = softmaxRows(scores)

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
