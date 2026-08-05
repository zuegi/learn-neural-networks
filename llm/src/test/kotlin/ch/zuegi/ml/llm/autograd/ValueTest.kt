package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.tanh

class ValueTest {


    †

    @Test
    fun `karpathy test`() {
        val a = Value(3.0)

        val b = a + a

        b.backward()
        assertThat(b.data).isEqualTo(6.0)
        assertThat(b.grad).isEqualTo(1.0)
        assertThat(a.data).isEqualTo(3.0)
        assertThat(a.grad).isEqualTo(2.0)
    }

    @Test
    fun `plus propagates gradient of one to both operands`() {
        val a = Value(2.0)
        val b = Value(3.0)

        val c = a + b
        c.backward()

        assertThat(c.data).isEqualTo(5.0)
        assertThat(a.grad).isEqualTo(1.0)
        assertThat(b.grad).isEqualTo(1.0)
    }

    @Test
    fun `times uses product rule`() {
        val a = Value(2.0)
        val b = Value(3.0)

        val c = a * b
        c.backward()

        assertThat(c.data).isEqualTo(6.0)
        assertThat(a.grad).isEqualTo(3.0) // d(a*b)/da = b
        assertThat(b.grad).isEqualTo(2.0) // d(a*b)/db = a
    }

    @Test
    fun `shared node accumulates gradients from multiple paths`() {
        val a = Value(2.0)
        val b = Value(3.0)

        // d = a*b + a  -> ∂d/∂a = b + 1 = 4
        val d = a * b + a
        d.backward()

        assertThat(d.data).isEqualTo(8.0)
        assertThat(a.grad).isEqualTo(4.0)
        assertThat(b.grad).isEqualTo(2.0)
    }

    @Test
    fun `tanh derivative is one minus tanh squared`() {
        val x = Value(0.5)

        val y = x.tanh()
        y.backward()

        val expected = 1.0 - tanh(0.5) * tanh(0.5)
        assertThat(y.data).isCloseTo(tanh(0.5), offset(1e-12))
        assertThat(x.grad).isCloseTo(expected, offset(1e-12))
    }

    @Test
    fun `analytic gradient matches numeric gradient`() {
        // f(a, b) = tanh(a * b + a)
        fun forward(
            aVal: Double,
            bVal: Double,
        ): Double {
            val a = Value(aVal)
            val b = Value(bVal)
            return (a * b + a).tanh().data
        }

        val aVal = 0.3
        val bVal = -1.2

        // analytisch
        val a = Value(aVal)
        val b = Value(bVal)
        val out = (a * b + a).tanh()
        out.backward()

        // numerisch: (f(x+h) - f(x-h)) / 2h
        val h = 1e-6
        val numericGradA = (forward(aVal + h, bVal) - forward(aVal - h, bVal)) / (2 * h)
        val numericGradB = (forward(aVal, bVal + h) - forward(aVal, bVal - h)) / (2 * h)

        assertThat(a.grad).isCloseTo(numericGradA, offset(1e-6))
        assertThat(b.grad).isCloseTo(numericGradB, offset(1e-6))
    }
}
