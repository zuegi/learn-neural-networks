package ch.zuegi.ml.llm.autograd

import java.util.Random

/**
 * Trainierbare Multi-Head Self-Attention auf Basis von [Tensor].
 *
 * Fuehrt numHeads [SelfAttentionLayer] parallel aus, konkateniert deren
 * Outputs spaltenweise und projiziert via Wo zurueck auf embeddingDim.
 *
 *     input   [ctx, embeddingDim]
 *     head_h  [ctx, dK]                       // je Kopf
 *     concat  [ctx, numHeads * dK]
 *     output = concat · Wo  -> [ctx, embeddingDim]
 *
 * Die Rueckprojektion auf embeddingDim ermoeglicht spaeter Residual (input + output).
 *
 * @param embeddingDim Laenge der Input-Embeddings.
 * @param numHeads Anzahl paralleler Koepfe.
 * @param dK Dimension pro Kopf.
 * @param causal wenn true, maskiert jeder Kopf die Zukunft.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class MultiHeadAttentionLayer(
    private val embeddingDim: Int,
    private val numHeads: Int,
    private val dK: Int,
    causal: Boolean = false,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(numHeads > 0) { "numHeads muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    private val heads: List<SelfAttentionLayer> =
        (0 until numHeads).map { headIndex ->
            SelfAttentionLayer(
                embeddingDim = embeddingDim,
                dK = dK,
                causal = causal,
                seed = seed?.let { it + headIndex },
            )
        }

    // Wo: [numHeads * dK, embeddingDim], flach row-major
    val wOutput: Tensor =
        Tensor(DoubleArray(numHeads * dK * embeddingDim) { rnd.nextGaussian() * INIT_SCALE })

    /**
     * @param input Matrix-Tensor [ctx, embeddingDim], row-major.
     * @param ctx Anzahl Positionen.
     * @return Matrix-Tensor [ctx, embeddingDim], row-major.
     */
    fun forward(
        input: Tensor,
        ctx: Int,
    ): Tensor {
        val headOutputs = heads.map { it.forward(input, ctx) } // je [ctx, dK]
        val concat = Tensor.concatCols(headOutputs, ctx = ctx, colsEach = dK)
        return concat.matMul(wOutput, p = ctx, q = numHeads * dK, r = embeddingDim)
    }

    fun parameters(): List<Tensor> = heads.flatMap { it.parameters() } + wOutput

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
