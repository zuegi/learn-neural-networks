package ch.zuegi.ml.llm.kapitel4.library.autograd

class TransformerBlockMultikTensor(
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

    private val attentionNorm = LayerNormMultikTensor(embeddingDim)
    private val attention =
        MultiHeadAttentionMultikTensor(
            embeddingDim = embeddingDim,
            numHeads = numHeads,
            dK = dK,
            causal = causal,
            dropoutProb = dropoutProb,
            seed = seed,
        )

    private val feedForwardNorm = LayerNormMultikTensor(embeddingDim)
    private val feedForward =
        FeedForwardMultikTensor(
            embeddingDim = embeddingDim,
            hiddenDim = hiddenDim,
            seed = seed?.let { it + FEED_FORWARD_SEED_OFFSET },
        )

    fun forward(
        input: TensorMultik,
        ctx: Int,
        training: Boolean = false,
    ): TensorMultik {
        val normedForAttention = perRow(input, ctx) { attentionNorm.forward(it) }
        val attended = input + attention.forward(normedForAttention, ctx, training)

        val normedForFeedForward = perRow(attended, ctx) { feedForwardNorm.forward(it) }
        return attended + perRow(normedForFeedForward, ctx) { feedForward.forward(it) }
    }

    fun parameters(): List<TensorMultik> =
        attentionNorm.parameters() +
            attention.parameters() +
            feedForwardNorm.parameters() +
            feedForward.parameters()

    private fun perRow(
        matrix: TensorMultik,
        ctx: Int,
        block: (TensorMultik) -> TensorMultik,
    ): TensorMultik {
        val rows = (0 until ctx).map { block(matrix.row(it, embeddingDim)) }
        return TensorMultik.stackRows(rows, embeddingDim)
    }

    companion object {
        private const val FEED_FORWARD_SEED_OFFSET = 1000L
    }
}
