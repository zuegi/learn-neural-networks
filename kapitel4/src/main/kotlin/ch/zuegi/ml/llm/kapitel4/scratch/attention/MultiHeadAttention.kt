package ch.zuegi.ml.llm.kapitel4.scratch.attention

import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Multi-Head Causal Self-Attention nach Raschka.
 *
 * Eine gemeinsame W_q/W_k/W_v-Projektion wird intern in numHeads Köpfe gesplittet.
 *
 * Ablauf:
 *   Q = X * Wq   [ctx, numHeads * dK]
 *   K = X * Wk   [ctx, numHeads * dK]
 *   V = X * Wv   [ctx, numHeads * dK]
 *   pro Kopf h:
 *     scores_h = Q_h * K_h^T / sqrt(dK)
 *     scores_h = causalMask(scores_h)
 *     weights_h = softmax(scores_h)
 *     weights_h = dropout(weights_h)  // nur training
 *     head_h = weights_h * V_h
 *   concat = [head_0 | ... | head_n]  [ctx, numHeads * dK]
 *   output = concat * Wo              [ctx, embeddingDim]
 */
class MultiHeadAttention(
    private val embeddingDim: Int,
    private val numHeads: Int,
    private val dK: Int,
    private val dropoutProb: Double = 0.0,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(numHeads > 0) { "numHeads muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
        require(dropoutProb in 0.0..1.0) { "dropoutProb muss in [0.0, 1.0] liegen" }
    }

    private val headDim = numHeads * dK
    private val rnd = if (seed != null) Random(seed) else Random()

    val wQuery: Array<DoubleArray> = randomMatrix(embeddingDim, headDim)
    val wKey: Array<DoubleArray> = randomMatrix(embeddingDim, headDim)
    val wValue: Array<DoubleArray> = randomMatrix(embeddingDim, headDim)
    val wOutput: Array<DoubleArray> = randomMatrix(headDim, embeddingDim)

    fun forward(
        input: Array<DoubleArray>,
        training: Boolean = false,
    ): Array<DoubleArray> {
        require(input.isNotEmpty()) { "input darf nicht leer sein" }
        require(input[0].size == embeddingDim) {
            "Input-Embedding-Dim ${input[0].size} passt nicht zu embeddingDim $embeddingDim"
        }

        val query = matMul(input, wQuery) // [ctx, headDim]
        val key = matMul(input, wKey)
        val value = matMul(input, wValue)

        val concat = processHeads(query, key, value, training) // [ctx, headDim]
        return matMul(concat, wOutput) // [ctx, embeddingDim]
    }

    private fun processHeads(
        query: Array<DoubleArray>,
        key: Array<DoubleArray>,
        value: Array<DoubleArray>,
        training: Boolean,
    ): Array<DoubleArray> {
        val ctx = query.size
        val result = Array(ctx) { DoubleArray(headDim) }

        for (h in 0 until numHeads) {
            val start = h * dK
            val end = start + dK

            val qH = sliceCols(query, start, end)
            val kH = sliceCols(key, start, end)
            val vH = sliceCols(value, start, end)

            val scores = scaledScores(qH, kH)
            maskCausal(scores)

            var weights = softmaxRows(scores)
            if (training && dropoutProb > 0.0) {
                weights = applyDropout(weights, dropoutProb)
            }

            val headOutput = matMul(weights, vH) // [ctx, dK]

            for (pos in 0 until ctx) {
                for (d in 0 until dK) {
                    result[pos][start + d] = headOutput[pos][d]
                }
            }
        }

        return result
    }

    private fun sliceCols(
        matrix: Array<DoubleArray>,
        from: Int,
        until: Int,
    ): Array<DoubleArray> =
        Array(matrix.size) { i ->
            DoubleArray(until - from) { d -> matrix[i][from + d] }
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

    private fun applyDropout(
        weights: Array<DoubleArray>,
        p: Double,
    ): Array<DoubleArray> {
        if (p == 1.0) return Array(weights.size) { DoubleArray(weights[it].size) }
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
