package ch.zuegi.ml.llm.kapitel3.library

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import java.util.Random
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Single-Head Causal Self-Attention (nur Forward-Pass) mit optionalem Attention-Dropout.
 *
 * Position i darf nur auf Positionen j <= i schauen.
 */
class CausalAttentionMultik(
    private val embeddingDim: Int,
    private val dK: Int,
    private val dropoutProb: Double = 0.0,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
        require(dropoutProb in 0.0..1.0) { "dropoutProb muss in [0.0, 1.0] liegen" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val wQuery: NDArray<Double, D2> = randomMatrix(embeddingDim, dK)
    val wKey: NDArray<Double, D2> = randomMatrix(embeddingDim, dK)
    val wValue: NDArray<Double, D2> = randomMatrix(embeddingDim, dK)

    /**
     * @param input Matrix [contextLength, embeddingDim]
     * @param training true = Dropout aktiv, false = kein Dropout
     */
    fun forward(
        input: NDArray<Double, D2>,
        training: Boolean = false,
    ): NDArray<Double, D2> {
        require(input.shape[0] > 0) { "input darf nicht leer sein" }
        require(input.shape[1] == embeddingDim) {
            "Input-Embedding-Dim ${input.shape[1]} passt nicht zu embeddingDim $embeddingDim"
        }

        val query = matMul(input, wQuery)
        val key = matMul(input, wKey)
        val value = matMul(input, wValue)

        val scores = scaledScores(query, key)
        val maskedScores = maskCausal(scores)

        var weights = softmaxRows(maskedScores)
        if (training && dropoutProb > 0.0) {
            weights = applyDropout(weights, dropoutProb)
        }

        return matMul(weights, value)
    }

    private fun scaledScores(
        query: NDArray<Double, D2>,
        key: NDArray<Double, D2>,
    ): NDArray<Double, D2> {
        val scale = sqrt(dK.toDouble())
        val ctx = query.shape[0]

        return mk.ndarray(
            List(ctx) { i ->
                List(ctx) { j ->
                    var sum = 0.0
                    for (d in 0 until dK) {
                        sum += query[i][d] * key[j][d]
                    }
                    sum / scale
                }
            },
        )
    }

    private fun maskCausal(scores: NDArray<Double, D2>): NDArray<Double, D2> {
        val rows = scores.shape[0]
        val cols = scores.shape[1]

        return mk.ndarray(
            List(rows) { i ->
                List(cols) { j ->
                    if (j > i) Double.NEGATIVE_INFINITY else scores[i][j]
                }
            },
        )
    }

    private fun softmaxRows(scores: NDArray<Double, D2>): NDArray<Double, D2> {
        val rows = scores.shape[0]
        val cols = scores.shape[1]

        return mk.ndarray(
            List(rows) { i ->
                val max = (0 until cols).maxOf { scores[i][it] }
                val exps = DoubleArray(cols) { exp(scores[i][it] - max) }
                val sum = exps.sum()
                List(cols) { exps[it] / sum }
            },
        )
    }

    private fun applyDropout(
        weights: NDArray<Double, D2>,
        p: Double,
    ): NDArray<Double, D2> {
        if (p == 0.0) return weights

        val rows = weights.shape[0]
        val cols = weights.shape[1]

        if (p == 1.0) {
            return mk.ndarray(List(rows) { List(cols) { 0.0 } })
        }

        val keepScale = 1.0 / (1.0 - p)
        return mk.ndarray(
            List(rows) { i ->
                List(cols) { j ->
                    if (rnd.nextDouble() < p) 0.0 else weights[i][j] * keepScale
                }
            },
        )
    }

    private fun matMul(
        a: NDArray<Double, D2>,
        b: NDArray<Double, D2>,
    ): NDArray<Double, D2> {
        require(a.shape[1] == b.shape[0]) {
            "Matrix-Dimensionen passen nicht: [${a.shape[0]}, ${a.shape[1]}] x [${b.shape[0]}, ${b.shape[1]}]"
        }

        val rows = a.shape[0]
        val inner = a.shape[1]
        val cols = b.shape[1]

        return mk.ndarray(
            List(rows) { i ->
                List(cols) { j ->
                    var sum = 0.0
                    for (k in 0 until inner) {
                        sum += a[i][k] * b[k][j]
                    }
                    sum
                }
            },
        )
    }

    private fun randomMatrix(
        rows: Int,
        cols: Int,
    ): NDArray<Double, D2> =
        mk.ndarray(
            List(rows) {
                List(cols) { rnd.nextGaussian() * INIT_SCALE }
            },
        )

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
