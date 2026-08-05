package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.tanh

class TensorTest {
    @Test
    fun `elementwise plus computes and backprops`() {
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0))
        val b = Tensor(doubleArrayOf(4.0, 5.0, 6.0))

        val c = a + b
        c.backward()

        assertThat(c.data).containsExactly(5.0, 7.0, 9.0)
        assertThat(a.grad).containsExactly(1.0, 1.0, 1.0)
        assertThat(b.grad).containsExactly(1.0, 1.0, 1.0)
    }

    @Test
    fun `elementwise times uses product rule per element`() {
        val a = Tensor(doubleArrayOf(2.0, 3.0))
        val b = Tensor(doubleArrayOf(5.0, 7.0))

        val c = a * b
        c.backward()

        assertThat(c.data).containsExactly(10.0, 21.0)
        assertThat(a.grad).containsExactly(5.0, 7.0) // = b
        assertThat(b.grad).containsExactly(2.0, 3.0) // = a
    }

    @Test
    fun `tanh derivative per element`() {
        val x = Tensor(doubleArrayOf(0.5, -0.3))

        val y = x.tanh()
        y.backward()

        for (i in x.data.indices) {
            val expected = 1.0 - tanh(x.data[i]) * tanh(x.data[i])
            assertThat(y.data[i]).isCloseTo(tanh(x.data[i]), offset(1e-12))
            assertThat(x.grad[i]).isCloseTo(expected, offset(1e-12))
        }
    }

    @Test
    fun `shared tensor accumulates gradients`() {
        val a = Tensor(doubleArrayOf(2.0, 3.0))
        val b = Tensor(doubleArrayOf(4.0, 5.0))

        // d = a*b + a
        val d = a * b + a
        d.backward()

        // ∂d/∂a = b + 1, ∂d/∂b = a
        assertThat(a.grad).containsExactly(5.0, 6.0) // b + 1
        assertThat(b.grad).containsExactly(2.0, 3.0) // a
    }

    @Test
    fun `analytic gradient matches numeric gradient`() {
        // f(x) = tanh(x * w + b), Summe der Ausgaben
        val xVal = doubleArrayOf(0.3, -0.7)
        val wVal = doubleArrayOf(1.2, -0.5)
        val bVal = doubleArrayOf(0.1, 0.4)

        fun forwardSum(
            x: DoubleArray,
            w: DoubleArray,
            b: DoubleArray,
        ): Double {
            val out = DoubleArray(x.size) { tanh(x[it] * w[it] + b[it]) }
            return out.sum()
        }

        // analytisch
        val x = Tensor(xVal)
        val w = Tensor(wVal)
        val b = Tensor(bVal)
        val out = (x * w + b).tanh()
        out.backward()

        // numerisch: partielle Ableitung nach jedem x[i]
        val h = 1e-6
        for (i in xVal.indices) {
            val plus = xVal.copyOf().also { it[i] += h }
            val minus = xVal.copyOf().also { it[i] -= h }
            val numericGrad = (forwardSum(plus, wVal, bVal) - forwardSum(minus, wVal, bVal)) / (2 * h)

            assertThat(x.grad[i]).isCloseTo(numericGrad, offset(1e-6))
        }
    }
}
