package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

class TransformerBlockMultik(
    private val embeddingDim: Int,
    numHeads: Int,
    dK: Int,
    hiddenDim: Int = 4 * embeddingDim,
    causal: Boolean = false,
    dropoutProb: Double = 0.0,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(numHeads > 0) { "numHeads muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
    }

    private val attentionNorm = LayerNormMultik(embeddingDim)
    private val attention =
        MultiHeadAttentionMultik(
            embeddingDim = embeddingDim,
            numHeads = numHeads,
            dK = dK,
            causal = causal,
            dropoutProb = dropoutProb,
            seed = seed,
        )

    private val feedForwardNorm = LayerNormMultik(embeddingDim)
    private val feedForward =
        FeedForwardMultik(
            embeddingDim = embeddingDim,
            hiddenDim = hiddenDim,
            seed = seed?.let { it + FEED_FORWARD_SEED_OFFSET },
        )

    fun forward(
        input: NDArray<Double, D2>,
        training: Boolean = false,
    ): NDArray<Double, D2> {
        val normedForAttention = perRow(input) { attentionNorm.forward(it) }
        val attended = addMatrices(input, attention.forward(normedForAttention, training))

        val normedForFeedForward = perRow(attended) { feedForwardNorm.forward(it) }
        return addMatrices(attended, perRow(normedForFeedForward) { feedForward.forward(it) })
    }

    private fun perRow(
        matrix: NDArray<Double, D2>,
        block: (NDArray<Double, D1>) -> NDArray<Double, D1>,
    ): NDArray<Double, D2> {
        val rows = matrix.shape[0]
        return mk.ndarray(
            List(rows) { r ->
                block(row(matrix, r)).toList()
            },
        )
    }

    private fun row(
        matrix: NDArray<Double, D2>,
        rowIndex: Int,
    ): NDArray<Double, D1> = mk.ndarray(DoubleArray(matrix.shape[1]) { c -> matrix[rowIndex][c] })

    private fun addMatrices(
        a: NDArray<Double, D2>,
        b: NDArray<Double, D2>,
    ): NDArray<Double, D2> {
        require(a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1]) {
            "Matrix-Dimensionen muessen gleich sein"
        }
        return mk.ndarray(
            List(a.shape[0]) { r ->
                List(a.shape[1]) { c -> a[r][c] + b[r][c] }
            },
        )
    }

    companion object {
        private const val FEED_FORWARD_SEED_OFFSET = 1000L
    }
}
