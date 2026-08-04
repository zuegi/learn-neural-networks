package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test

class FeedForwardTest {
    private fun identityInput(
        ctx: Int,
        dim: Int,
    ): Array<DoubleArray> = Array(ctx) { i -> DoubleArray(dim) { d -> if (d == i) 1.0 else 0.0 } }

    @Test
    fun `output has same shape as input`() {
        val ff = FeedForward(embeddingDim = 4, seed = 42)
        val input = identityInput(ctx = 3, dim = 4)

        val output = ff.forward(input)

        assertThat(output.size).isEqualTo(3)
        assertThat(output.all { it.size == 4 }).isTrue()
    }

    @Test
    fun `hidden dimension defaults to four times embeddingDim`() {
        val embeddingDim = 4
        val ff = FeedForward(embeddingDim, seed = 42)

        assertThat(ff.w1[0].size).isEqualTo(4 * embeddingDim)
        assertThat(ff.w2.size).isEqualTo(4 * embeddingDim)
    }

    @Test
    fun `zero input yields bias-only output`() {
        val embeddingDim = 3
        val ff = FeedForward(embeddingDim, seed = 42)

        // b1 und b2 sind 0 -> gelu(0)=0 -> Output = b2 = 0
        val input = arrayOf(DoubleArray(embeddingDim) { 0.0 })
        val output = ff.forward(input)

        for (value in output[0]) {
            assertThat(value).isCloseTo(0.0, offset(1e-12))
        }
    }

    @Test
    fun `forward matches manual computation for one row`() {
        val embeddingDim = 3
        val ff = FeedForward(embeddingDim, hiddenDim = 5, seed = 42)
        val x = doubleArrayOf(0.1, -0.2, 0.3)

        val output = ff.forward(arrayOf(x))

        // manuell: h = gelu(x·W1 + b1), out = h·W2 + b2
        val hidden =
            DoubleArray(5) { j ->
                var sum = ff.b1[j]
                for (k in x.indices) sum += x[k] * ff.w1[k][j]
                gelu(sum)
            }
        val expected =
            DoubleArray(embeddingDim) { j ->
                var sum = ff.b2[j]
                for (k in hidden.indices) sum += hidden[k] * ff.w2[k][j]
                sum
            }

        assertThat(output[0]).containsExactly(*expected)
    }

    private fun gelu(x: Double): Double {
        val c = kotlin.math.sqrt(2.0 / kotlin.math.PI)
        val inner = c * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + kotlin.math.tanh(inner))
    }
}
