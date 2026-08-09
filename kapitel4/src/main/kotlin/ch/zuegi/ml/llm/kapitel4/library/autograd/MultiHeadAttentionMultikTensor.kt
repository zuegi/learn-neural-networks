package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import java.util.Random
import kotlin.math.sqrt

/**
 * Trainierbare Multi-Head Self-Attention auf Basis von [TensorMultik].
 *
 * Raschka-Stil: gemeinsame Wq/Wk/Wv, intern in [numHeads] Köpfe gesplittet.
 * Optional causal maskiert, mit Attention-Dropout.
 *
 * @param embeddingDim Länge eines Token-Vektors.
 * @param numHeads Anzahl paralleler Attention-Köpfe.
 * @param dK Dimension pro Kopf.
 * @param causal wenn true, wird Zukunft maskiert.
 * @param dropoutProb Attention-Dropout auf den Softmax-Gewichten.
 * @param seed optionaler Seed für reproduzierbare Initialisierung.
 */
class MultiHeadAttentionMultikTensor(
    private val embeddingDim: Int,
    private val numHeads: Int,
    private val dK: Int,
    private val causal: Boolean = false,
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

    val wQuery: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(embeddingDim * headDim) { rnd.nextGaussian() * INIT_SCALE }),
        )
    val wKey: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(embeddingDim * headDim) { rnd.nextGaussian() * INIT_SCALE }),
        )
    val wValue: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(embeddingDim * headDim) { rnd.nextGaussian() * INIT_SCALE }),
        )
    val wOutput: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(headDim * embeddingDim) { rnd.nextGaussian() * INIT_SCALE }),
        )

    /**
     * @param input Matrix-Tensor [ctx, embeddingDim], row-major.
     * @param ctx Anzahl Positionen.
     * @param training true = Attention-Dropout aktiv.
     * @return Matrix-Tensor [ctx, embeddingDim], row-major.
     */
    fun forward(
        input: TensorMultik,
        ctx: Int,
        training: Boolean = false,
    ): TensorMultik {
        require(input.size == ctx * embeddingDim) {
            "input.size ${input.size} passt nicht zu ctx*embeddingDim=${ctx * embeddingDim}"
        }

        val query = input.matMul(wQuery, p = ctx, q = embeddingDim, r = headDim)
        val key = input.matMul(wKey, p = ctx, q = embeddingDim, r = headDim)
        val value = input.matMul(wValue, p = ctx, q = embeddingDim, r = headDim)

        val headOutputs =
            (0 until numHeads).map { h ->
                val fromCol = h * dK
                forwardHead(
                    query = query.sliceCols(ctx, headDim, fromCol, dK),
                    key = key.sliceCols(ctx, headDim, fromCol, dK),
                    value = value.sliceCols(ctx, headDim, fromCol, dK),
                    ctx = ctx,
                    training = training,
                )
            }

        val concatenated = TensorMultik.concatCols(headOutputs, ctx = ctx, colsEach = dK)
        return concatenated.matMul(wOutput, p = ctx, q = headDim, r = embeddingDim)
    }

    fun parameters(): List<TensorMultik> = listOf(wQuery, wKey, wValue, wOutput)

    private fun forwardHead(
        query: TensorMultik,
        key: TensorMultik,
        value: TensorMultik,
        ctx: Int,
        training: Boolean,
    ): TensorMultik {
        val scale = sqrt(dK.toDouble())

        val rows =
            (0 until ctx).map { rowIndex ->
                val queryRow = query.row(rowIndex, dK)
                var scores = key.dotWithRows(queryRow, rows = ctx, cols = dK)
                scores = if (causal) scores.maskCausalScale(rowIndex, scale) else scores.scale(scale)

                var weights = scores.softmax()
                if (training && dropoutProb > 0.0) {
                    weights = weights.dropout(dropoutProb, rnd)
                }

                weightedValueSum(weights, value, ctx)
            }

        return TensorMultik.stackRows(rows, dK)
    }

    private fun weightedValueSum(
        weights: TensorMultik,
        value: TensorMultik,
        ctx: Int,
    ): TensorMultik {
        var acc = weights.row(0, 1).broadcastMul(value.row(0, dK))
        for (i in 1 until ctx) {
            acc = acc + weights.row(i, 1).broadcastMul(value.row(i, dK))
        }
        return acc
    }

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
