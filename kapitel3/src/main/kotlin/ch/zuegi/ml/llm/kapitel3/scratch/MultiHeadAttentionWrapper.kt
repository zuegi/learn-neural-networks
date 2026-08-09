package ch.zuegi.ml.llm.kapitel3.scratch

import java.util.Random

class MultiHeadAttentionWrapper(
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
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    private val heads: List<ch.zuegi.ml.llm.kapitel3.scratch.CausalAttention> =
        (0 until numHeads).map { headIndex ->
            CausalAttention(
                embeddingDim = embeddingDim,
                dK = dK,
                dropoutProb = dropoutProb,
                seed = seed?.let { it + headIndex },
            )
        }

    val wOutput: Array<DoubleArray> =
        Array(numHeads * dK) { DoubleArray(embeddingDim) { rnd.nextGaussian() * INIT_SCALE } }

    fun forward(
        input: Array<DoubleArray>,
        training: Boolean = false,
    ): Array<DoubleArray> {
        val concatenated = concatHeads(input, training)
        return matMul(concatenated, wOutput)
    }

    internal fun concatHeads(
        input: Array<DoubleArray>,
        training: Boolean = false,
    ): Array<DoubleArray> {
        val headOutputs = heads.map { it.forward(input, training) }
        val contextLength = input.size
        val concatDim = numHeads * dK

        return Array(contextLength) { pos ->
            val row = DoubleArray(concatDim)
            var offset = 0
            for (headOutput in headOutputs) {
                val headRow = headOutput[pos]
                for (d in headRow.indices) {
                    row[offset + d] = headRow[d]
                }
                offset += dK
            }
            row
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

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
