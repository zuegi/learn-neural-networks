package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class SelfAttentionTest {
    private fun identityInput(
        ctx: Int,
        dim: Int,
    ): Array<DoubleArray> = Array(ctx) { i -> DoubleArray(dim) { d -> if (d == i) 1.0 else 0.0 } }

    @Test
    fun `forward output has shape contextLength x dK`() {
        val attention = SelfAttention(embeddingDim = 4, dK = 3, seed = 42)
        val input = identityInput(ctx = 4, dim = 4)

        val output = attention.forward(input)

        assertThat(output.size).isEqualTo(4)
        assertThat(output.all { it.size == 3 }).isTrue()
    }

    @Test
    fun `causal attention first position equals its own value row`() {
        val embeddingDim = 4
        val dK = 3
        val attention = SelfAttention(embeddingDim, dK, causal = true, seed = 42)
        val input = identityInput(ctx = 4, dim = embeddingDim)

        val output = attention.forward(input)

        // Position 0 darf nur auf sich selbst schauen -> Output[0] == V[0]
        val expectedValueRow0 = matMulRow(input[0], attention.wValue)
        assertThat(output[0]).containsExactly(*expectedValueRow0)
    }

    @Test
    fun `causal and non-causal differ for early positions but match at last`() {
        val embeddingDim = 4
        val dK = 3
        val input = identityInput(ctx = 4, dim = embeddingDim)

        val causal = SelfAttention(embeddingDim, dK, causal = true, seed = 42)
        val full = SelfAttention(embeddingDim, dK, causal = false, seed = 42)

        val causalOut = causal.forward(input)
        val fullOut = full.forward(input)

        // Letzte Position sieht in beiden Faellen die ganze Sequenz -> identisch
        val last = input.size - 1
        assertThat(causalOut[last]).containsExactly(*fullOut[last])

        // Fruehe Position unterscheidet sich (causal maskiert Zukunft)
        assertThat(causalOut[0]).isNotEqualTo(fullOut[0])
    }

    private fun matMulRow(
        row: DoubleArray,
        matrix: Array<DoubleArray>,
    ): DoubleArray {
        val cols = matrix[0].size
        return DoubleArray(cols) { j ->
            var sum = 0.0
            for (k in row.indices) {
                sum += row[k] * matrix[k][j]
            }
            sum
        }
    }
}
