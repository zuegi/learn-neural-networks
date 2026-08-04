package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class MultiHeadAttentionTest {
    private fun identityInput(
        ctx: Int,
        dim: Int,
    ): Array<DoubleArray> = Array(ctx) { i -> DoubleArray(dim) { d -> if (d == i) 1.0 else 0.0 } }

    @Test
    fun `forward output has shape contextLength x embeddingDim`() {
        val mha = MultiHeadAttention(embeddingDim = 4, numHeads = 8, dK = 8, seed = 42)
        val input = identityInput(ctx = 4, dim = 4)

        val output = mha.forward(input)

        assertThat(output.size).isEqualTo(4)
        assertThat(output.all { it.size == 4 }).isTrue() // embeddingDim, nicht numHeads*dK
    }

    @Test
    fun `heads produce different values`() {
        val numHeads = 2
        val dK = 3
        val mha = MultiHeadAttention(embeddingDim = 4, numHeads = numHeads, dK = dK, seed = 42)
        val input = identityInput(ctx = 4, dim = 4)

        val concat = mha.concatHeads(input)

        val head0Block = concat[0].sliceArray(0 until dK)
        val head1Block = concat[0].sliceArray(dK until 2 * dK)

        assertThat(head0Block).isNotEqualTo(head1Block)
    }

    @Test
    fun `causal concat first position only sees itself per head`() {
        val numHeads = 2
        val dK = 3
        val embeddingDim = 4
        val mha = MultiHeadAttention(embeddingDim, numHeads, dK, causal = true, seed = 42)
        val input = identityInput(ctx = 4, dim = embeddingDim)

        val concat = mha.concatHeads(input)

        val expected = DoubleArray(numHeads * dK)
        var offset = 0
        for (headIndex in 0 until numHeads) {
            val head = SelfAttention(embeddingDim, dK, causal = true, seed = 42L + headIndex)
            val headRow0 = head.forward(input)[0]
            for (d in headRow0.indices) {
                expected[offset + d] = headRow0[d]
            }
            offset += dK
        }

        assertThat(concat[0]).containsExactly(*expected)
    }

    @Test
    fun `forward applies output projection to embeddingDim`() {
        val numHeads = 2
        val dK = 3
        val embeddingDim = 4
        val mha = MultiHeadAttention(embeddingDim, numHeads, dK, seed = 42)
        val input = identityInput(ctx = 4, dim = embeddingDim)

        val concat = mha.concatHeads(input)
        val output = mha.forward(input)

        // output[pos] muss = concat[pos] · wOutput sein
        val expectedRow0 =
            DoubleArray(embeddingDim) { j ->
                var sum = 0.0
                for (k in concat[0].indices) {
                    sum += concat[0][k] * mha.wOutput[k][j]
                }
                sum
            }

        assertThat(output[0]).containsExactly(*expectedRow0)
    }
}
