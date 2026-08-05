package ch.zuegi.ml.llm.autograd

/**
 * Trainierbarer Transformer-Block auf Basis von [Tensor].
 *
 * Kombiniert [MultiHeadAttentionLayer] und [FeedForwardLayer], jeweils mit
 * vorgeschalteter [LayerNormLayer] (Pre-LN) und Residual-Verbindung.
 *
 *     a   = x + MultiHeadAttention(LayerNorm1(x))
 *     out = a + FeedForward(LayerNorm2(a))
 *
 * LayerNorm und FeedForward wirken pro Token-Zeile; Attention und Residual
 * wirken auf der ganzen Matrix [ctx, embeddingDim]. Input- und Output-Form sind
 * identisch, sodass Bloecke gestapelt werden koennen.
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param numHeads Anzahl paralleler Attention-Koepfe.
 * @param dK Dimension pro Kopf.
 * @param hiddenDim versteckte Dimension des Feed-Forward-Netzes, Standard 4 * embeddingDim.
 * @param causal wenn true, maskiert die Attention die Zukunft.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class TransformerBlockLayer(
    private val embeddingDim: Int,
    numHeads: Int,
    dK: Int,
    hiddenDim: Int = 4 * embeddingDim,
    causal: Boolean = false,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(numHeads > 0) { "numHeads muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
    }

    private val attentionNorm = LayerNormLayer(embeddingDim)
    private val attention =
        MultiHeadAttentionLayer(
            embeddingDim = embeddingDim,
            numHeads = numHeads,
            dK = dK,
            causal = causal,
            seed = seed,
        )

    private val feedForwardNorm = LayerNormLayer(embeddingDim)
    private val feedForward =
        FeedForwardLayer(
            embeddingDim = embeddingDim,
            hiddenDim = hiddenDim,
            seed = seed?.let { it + FEED_FORWARD_SEED_OFFSET },
        )

    /**
     * @param input Matrix-Tensor [ctx, embeddingDim], row-major.
     * @param ctx Anzahl Positionen.
     * @return Matrix-Tensor gleicher Form [ctx, embeddingDim].
     */
    fun forward(
        input: Tensor,
        ctx: Int,
    ): Tensor {
        val normedForAttention = perRow(input, ctx) { attentionNorm.forward(it) }
        val attended = input + attention.forward(normedForAttention, ctx)

        val normedForFeedForward = perRow(attended, ctx) { feedForwardNorm.forward(it) }
        return attended + perRow(normedForFeedForward, ctx) { feedForward.forward(it) }
    }

    fun parameters(): List<Tensor> =
        attentionNorm.parameters() +
            attention.parameters() +
            feedForwardNorm.parameters() +
            feedForward.parameters()

    /**
     * Wendet ein zeilenweises Modul auf jede Zeile der Matrix [ctx, embeddingDim]
     * an und stapelt die Ergebnisse wieder zu einer Matrix.
     */
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
