package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.sqrt
import kotlin.math.tanh

class TensorGeluTest {
    private fun gelu(x: Double): Double {
        val c = sqrt(2.0 / PI)
        val inner = c * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + tanh(inner))
    }

    @Test
    fun `gelu computes tanh approximation`() {
        val x = Tensor(doubleArrayOf(-1.0, 0.0, 1.0, 2.0))

        val y = x.gelu()

        for (i in x.data.indices) {
            assertThat(y.data[i]).isCloseTo(gelu(x.data[i]), offset(1e-12))
        }
    }

    @Test
    fun `gelu of zero is zero`() {
        val y = Tensor(doubleArrayOf(0.0)).gelu()

        assertThat(y.data[0]).isCloseTo(0.0, offset(1e-12))
    }

    @Test
    fun `gelu gradient matches numeric`() {
        val xVal = doubleArrayOf(-1.5, -0.3, 0.7, 2.1)

        val x = Tensor(xVal)
        x.gelu().backward()

        val h = 1e-6
        for (i in xVal.indices) {
            val numeric = (gelu(xVal[i] + h) - gelu(xVal[i] - h)) / (2 * h)
            assertThat(x.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }
}

