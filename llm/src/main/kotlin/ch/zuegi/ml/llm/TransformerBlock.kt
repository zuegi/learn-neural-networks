package ch.zuegi.ml.llm

/**
 * Transformer-Block (nur Forward-Pass).
 *
 * Kombiniert Multi-Head-Attention und ein Feed-Forward-Netz, jeweils mit
 * vorgeschalteter Layer Normalization (Pre-LN) und Residual-Verbindung.
 *
 * Ablauf (Pre-LN):
 *
 *     a   = x + MultiHeadAttention(LayerNorm1(x))
 *     out = a + FeedForward(LayerNorm2(a))
 *
 * Input und Output haben dieselbe Form [contextLength, embeddingDim], sodass
 * mehrere Bloecke gestapelt werden koennen.
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param numHeads Anzahl paralleler Attention-Koepfe.
 * @param dK Dimension pro Kopf.
 * @param hiddenDim versteckte Dimension des Feed-Forward-Netzes, Standard 4 * embeddingDim.
 * @param causal wenn true, maskiert die Attention die Zukunft.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class TransformerBlock(
    embeddingDim: Int,
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

    private val attentionNorm = LayerNorm(embeddingDim)
    private val attention =
        MultiHeadAttention(
            embeddingDim = embeddingDim,
            numHeads = numHeads,
            dK = dK,
            causal = causal,
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
     * Forward-Pass des Blocks.
     *
     * @param input Embeddings der Form [contextLength, embeddingDim].
     * @return Output gleicher Form [contextLength, embeddingDim].
     */
    fun forward(input: Array<DoubleArray>): Array<DoubleArray> {
        val attended = addResidual(input, attention.forward(attentionNorm.forward(input)))
        return addResidual(attended, feedForward.forward(feedForwardNorm.forward(attended)))
    }

    private fun addResidual(
        residual: Array<DoubleArray>,
        subLayer: Array<DoubleArray>,
    ): Array<DoubleArray> =
        Array(residual.size) { i ->
            val a = residual[i]
            val b = subLayer[i]
            DoubleArray(a.size) { j -> a[j] + b[j] }
        }

    companion object {
        private const val FEED_FORWARD_SEED_OFFSET = 1000L
    }
}

