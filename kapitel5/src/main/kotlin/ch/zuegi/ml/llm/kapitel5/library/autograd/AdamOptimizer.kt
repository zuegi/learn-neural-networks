package ch.zuegi.ml.llm.kapitel5.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.math.pow

class AdamOptimizer(
    private val parameters: List<TensorMultik>,
    private val learningRate: Double = 0.001,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val eps: Double = 1e-8,
) {
    private val m = parameters.map { p -> mk.ndarray(DoubleArray(p.size)) }
    private val v = parameters.map { p -> mk.ndarray(DoubleArray(p.size)) }
    private var t = 0

    fun step() {
        t += 1
        for (i in parameters.indices) {
            val param = parameters[i]
            val grad = param.grad
            val mi = m[i]
            val vi = v[i]

            for (j in 0 until param.size) {
                val g = grad[j]

                mi[j] = beta1 * mi[j] + (1.0 - beta1) * g
                vi[j] = beta2 * vi[j] + (1.0 - beta2) * g * g

                val mHat = mi[j] / (1.0 - beta1.pow(t))
                val vHat = vi[j] / (1.0 - beta2.pow(t))

                param.data[j] -= learningRate * mHat / (kotlin.math.sqrt(vHat) + eps)
            }
        }
    }

    fun zeroGrad() {
        parameters.forEach { it.zeroGrad() }
    }
}
