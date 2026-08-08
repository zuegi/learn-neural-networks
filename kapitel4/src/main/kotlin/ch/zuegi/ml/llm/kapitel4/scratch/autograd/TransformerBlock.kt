package ch.zuegi.ml.llm.kapitel4.scratch.autograd

/**
 * Trainierbarer Transformer-Block auf Basis von [Tensor].
 *
 * Kombiniert [MultiHeadAttention] und [FeedForward], jeweils mit
 * vorgeschalteter [LayerNorm] (Pre-LN) und Residual-Verbindung.
 *
 *     a   = x + MultiHeadAttention(LayerNorm1(x))
 *     out = a + FeedForward(LayerNorm2(a))
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param numHeads Anzahl paralleler Attention-Koepfe.
 * @param dK Dimension pro Kopf.
 * @param hiddenDim versteckte Dimension des Feed-Forward-Netzes, Standard 4 * embeddingDim.
 * @param causal wenn true, maskiert die Attention die Zukunft.
 * @param dropoutProb Attention-Dropout auf den Softmax-Gewichten.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class TransformerBlock(
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

    private val attentionNorm = LayerNorm(embeddingDim)
    private val attention =
        MultiHeadAttention(
            embeddingDim = embeddingDim,
            numHeads = numHeads,
            dK = dK,
            causal = causal,
            dropoutProb = dropoutProb,
            seed = seed,
        )

    private val feedForwardNorm = LayerNorm(embeddingDim)
    private val feedForward =
        FeedForward(
            embeddingDim = embeddingDim,
            hiddenDim = hiddenDim,
            seed = seed?.let { it + FEED_FORWARD_SEED_OFFSET },
        )

    /**
     * @param input Matrix-Tensor [ctx, embeddingDim], row-major.
     * @param ctx Anzahl Positionen.
     * @param training true = Attention-Dropout aktiv.
     * @return Matrix-Tensor gleicher Form [ctx, embeddingDim].
     */
    fun forward(
        input: Tensor,
        ctx: Int,
        training: Boolean = false,
    ): Tensor {
        val normedForAttention = perRow(input, ctx) { attentionNorm.forward(it) }
        val attended = input + attention.forward(normedForAttention, ctx, training)

        val normedForFeedForward = perRow(attended, ctx) { feedForwardNorm.forward(it) }
        return attended + perRow(normedForFeedForward, ctx) { feedForward.forward(it) }
    }

    fun parameters(): List<Tensor> =
        attentionNorm.parameters() +
            attention.parameters() +
            feedForwardNorm.parameters() +
            feedForward.parameters()

    private fun perRow(
        matrix: Tensor,
        ctx: Int,
        block: (Tensor) -> Tensor,
    ): Tensor {
        val rows = (0 until ctx).map { block(matrix.row(it, embeddingDim)) }
        return Tensor.stackRows(rows, embeddingDim)
    }

    companion object {
        private const val FEED_FORWARD_SEED_OFFSET = 1000L
    }
}
