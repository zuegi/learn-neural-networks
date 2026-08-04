package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class TransformerBlockTest {
    private fun identityInput(
        ctx: Int,
        dim: Int,
    ): Array<DoubleArray> = Array(ctx) { i -> DoubleArray(dim) { d -> if (d == i) 1.0 else 0.0 } }

    @Test
    fun `output has same shape as input`() {
        val block = TransformerBlock(embeddingDim = 4, numHeads = 2, dK = 3, seed = 42)
        val input = identityInput(ctx = 4, dim = 4)

        val output = block.forward(input)

        assertThat(output.size).isEqualTo(4)
        assertThat(output.all { it.size == 4 }).isTrue()
    }

    @Test
    fun `output differs from input due to sublayers`() {
        val block = TransformerBlock(embeddingDim = 4, numHeads = 2, dK = 3, seed = 42)
        val input = identityInput(ctx = 4, dim = 4)

        val output = block.forward(input)

        assertThat(output[0]).isNotEqualTo(input[0])
    }

    @Test
    fun `same seed produces same output`() {
        val input = identityInput(ctx = 4, dim = 4)
        val a = TransformerBlock(embeddingDim = 4, numHeads = 2, dK = 3, seed = 42)
        val b = TransformerBlock(embeddingDim = 4, numHeads = 2, dK = 3, seed = 42)

        assertThat(a.forward(input).contentDeepEquals(b.forward(input))).isTrue()
    }
}

