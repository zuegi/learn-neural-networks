package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray

class LayerNormMultikTensor(
    private val embeddingDim: Int,
    private val eps: Double = 1e-5,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    val gamma: TensorMultik = TensorMultik(mk.ndarray(DoubleArray(embeddingDim) { 1.0 }))
    val beta: TensorMultik = TensorMultik(mk.ndarray(DoubleArray(embeddingDim) { 0.0 }))

    // tag::layer-norm-forward[]
    fun forward(x: TensorMultik): TensorMultik {
        require(x.size == embeddingDim) { "x.size ${x.size} passt nicht zu embeddingDim $embeddingDim" }
        return x.layerNorm(gamma, beta, eps)
    }
    // end::layer-norm-forward[]

    fun parameters(): List<TensorMultik> = listOf(gamma, beta)
}
