package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import kotlin.math.sqrt

class LayerNormMultikTensor(
    private val embeddingDim: Int,
    private val eps: Double = 1e-5,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    val gamma: TensorMultik =
        TensorMultik(
            org.jetbrains.kotlinx.multik.api.mk
                .ndarray(DoubleArray(embeddingDim) { 1.0 }),
        )
    val beta: TensorMultik =
        TensorMultik(
            org.jetbrains.kotlinx.multik.api.mk
                .ndarray(DoubleArray(embeddingDim) { 0.0 }),
        )

    fun forward(x: TensorMultik): TensorMultik {
        require(x.size == embeddingDim) { "x.size ${x.size} passt nicht zu embeddingDim $embeddingDim" }

        val mean = (0 until embeddingDim).sumOf { x.data[it] } / embeddingDim
        val variance =
            (0 until embeddingDim).sumOf { i ->
                val d = x.data[i] - mean
                d * d
            } / embeddingDim
        val std = sqrt(variance + eps)

        val xhat = DoubleArray(embeddingDim) { i -> (x.data[i] - mean) / std }
        val result = DoubleArray(embeddingDim) { i -> gamma.data[i] * xhat[i] + beta.data[i] }

        return TensorMultik(
            org.jetbrains.kotlinx.multik.api.mk
                .ndarray(result),
            listOf(x, gamma, beta),
        )
    }

    fun parameters(): List<TensorMultik> = listOf(gamma, beta)
}
