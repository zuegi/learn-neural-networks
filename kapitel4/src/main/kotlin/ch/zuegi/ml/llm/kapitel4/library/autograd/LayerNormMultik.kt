package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.math.sqrt

class LayerNormMultik(
    private val embeddingDim: Int,
    private val eps: Double = 1e-5,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    val gamma: NDArray<Double, D1> = mk.ndarray(DoubleArray(embeddingDim) { 1.0 })
    val beta: NDArray<Double, D1> = mk.ndarray(DoubleArray(embeddingDim) { 0.0 })

    fun forward(x: NDArray<Double, D1>): NDArray<Double, D1> {
        require(x.size == embeddingDim) { "x.size ${x.size} passt nicht zu embeddingDim $embeddingDim" }

        var mean = 0.0
        for (i in 0 until embeddingDim) mean += x[i]
        mean /= embeddingDim

        var variance = 0.0
        for (i in 0 until embeddingDim) {
            val d = x[i] - mean
            variance += d * d
        }
        variance /= embeddingDim

        val std = sqrt(variance + eps)

        val out =
            DoubleArray(embeddingDim) { i ->
                val xhat = (x[i] - mean) / std
                gamma[i] * xhat + beta[i]
            }
        return mk.ndarray(out)
    }
}
