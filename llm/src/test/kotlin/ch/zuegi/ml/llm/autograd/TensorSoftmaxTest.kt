package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.within
import org.junit.jupiter.api.Test

class TensorSoftmaxTest {
    @Test
    fun `output sums to one and is positive`() {
        val p = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 0.5)).softmax()

        assertThat(p.data.sum()).isCloseTo(1.0, within(1e-12))
        assertThat(p.data.all { it > 0.0 }).isTrue()
    }

    @Test
    fun `backward matches numeric gradient`() {
        val x = doubleArrayOf(0.3, -1.2, 0.8, 2.0)

        // Skalare Zielfunktion: f = Σ w_i * softmax(x)_i, feste Gewichte w
        val w = doubleArrayOf(0.5, -0.3, 1.0, 0.2)
        val n = x.size

        // analytisch
        val t = Tensor(x.copyOf())
        val p = t.softmax()
        val wTensor = Tensor(w.copyOf())
        val loss = (p * wTensor).matVecMul(Tensor(DoubleArray(n) { 1.0 }), m = 1, n = n)
        loss.backward()

        // numerisch
        val h = 1e-6

        fun f(v: DoubleArray): Double {
            val max = v.max()
            val exps = DoubleArray(n) { kotlin.math.exp(v[it] - max) }
            val s = exps.sum()
            var total = 0.0
            for (i in 0 until n) total += w[i] * (exps[i] / s)
            return total
        }

        for (i in 0 until n) {
            val plus = x.copyOf().also { it[i] += h }
            val minus = x.copyOf().also { it[i] -= h }
            val numeric = (f(plus) - f(minus)) / (2 * h)
            assertThat(t.grad[i]).isCloseTo(numeric, within(1e-6))
        }
    }
}
