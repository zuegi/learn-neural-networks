package ch.zuegi.ml.llm.autograd

import java.util.Random
import kotlin.math.sqrt

/**
 * Trainierbare Single-Head Self-Attention auf Basis von [Tensor].
 *
 * Eingabe: Matrix-Tensor [ctx, embeddingDim] (row-major). Ablauf:
 *
 *     Q = X · Wq ; K = X · Wk ; V = X · Wv          // je [ctx, dK]
 *     scores_i  = (K · Q_i) / sqrt(dK)              // Laenge ctx
 *     weights_i = softmax(maskCausal(scores_i))
 *     out_i     = Vᵀ · weights_i                    // [dK]
 *
 * Baut ausschliesslich auf gradient-gecheckten [Tensor]-Ops auf.
 *
 * @param embeddingDim Laenge der Input-Embeddings.
 * @param dK Dimension von Query/Key/Value.
 * @param causal wenn true, darf Position i nur auf j <= i schauen.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class SelfAttentionLayer(
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

    val wQuery: Tensor = randomMatrix(embeddingDim, dK)
    val wKey: Tensor = randomMatrix(embeddingDim, dK)
    val wValue: Tensor = randomMatrix(embeddingDim, dK)

    /**
     * @param input Matrix-Tensor [ctx, embeddingDim], row-major.
     * @param ctx Anzahl Positionen.
     * @return Matrix-Tensor [ctx, dK], row-major.
     */
    fun forward(
        input: Tensor,
        ctx: Int,
    ): Tensor {
        require(input.size == ctx * embeddingDim) {
            "input.size ${input.size} passt nicht zu ctx*embeddingDim=${ctx * embeddingDim}"
        }

        val query = input.matMul(wQuery, p = ctx, q = embeddingDim, r = dK)
        val key = input.matMul(wKey, p = ctx, q = embeddingDim, r = dK)
        val value = input.matMul(wValue, p = ctx, q = embeddingDim, r = dK)
        val valueT = value.transposeMatrix(rows = ctx, cols = dK) // [dK, ctx]

        val scale = sqrt(dK.toDouble())
        val outRows = ArrayList<Tensor>(ctx)

        for (i in 0 until ctx) {
            val queryRow = query.row(i, dK) // [dK]
            val scoresRaw = queryRow.matVecMul(key, m = ctx, n = dK) // K·Q_i -> [ctx]
            val scores =
                if (causal) {
                    scoresRaw.maskCausalScale(position = i, scale = scale)
                } else {
                    scoresRaw.scale(scale)
                }
            val weights = scores.softmax() // [ctx]
            outRows.add(weights.matVecMul(valueT, m = dK, n = ctx)) // Vᵀ·w -> [dK]
        }

        return Tensor.stackRows(outRows, dK)
    }

    fun parameters(): List<Tensor> = listOf(wQuery, wKey, wValue)

    private fun randomMatrix(
        rows: Int,
        cols: Int,
    ): Tensor = Tensor(DoubleArray(rows * cols) { rnd.nextGaussian() * INIT_SCALE })

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
