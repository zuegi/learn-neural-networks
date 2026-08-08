package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Test
import kotlin.math.abs

class MultiHeadAttentionTest {
    @Test
    fun `forward liefert output mit shape ctx mal embeddingDim`() {
        val model = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, seed = 42)
        val input = identityInput(ctx = 3, dim = 8)

        val output = model.forward(input, ctx = 3, training = false)

        assertEquals(3 * 8, output.size)
    }

    @Test
    fun `gleiches seed und eval modus erzeugen gleiche outputs`() {
        val m1 = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.3, seed = 7)
        val m2 = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.3, seed = 7)

        val input = identityInput(ctx = 3, dim = 8)

        val out1 = m1.forward(input, ctx = 3, training = false)
        val out2 = m2.forward(input, ctx = 3, training = false)

        assertArrayEquals(out1.data, out2.data, 1e-12)
    }

    @Test
    fun `dropout im training beeinflusst output`() {
        val noDropout = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.0, seed = 11)
        val withDropout = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.5, seed = 11)

        val input = identityInput(ctx = 3, dim = 8)

        val out1 = noDropout.forward(input, ctx = 3, training = true)
        val out2 = withDropout.forward(input, ctx = 3, training = true)

        assertFalse(out1.data.zip(out2.data.toList()).all { (a, b) -> abs(a - b) < 1e-12 })
    }

    @Test
    fun `causal und non-causal liefern unterschiedliche outputs`() {
        val nonCausal = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, causal = false, seed = 99)
        val causal = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, causal = true, seed = 99)

        val input = identityInput(ctx = 3, dim = 8)

        val outNonCausal = nonCausal.forward(input, ctx = 3, training = false)
        val outCausal = causal.forward(input, ctx = 3, training = false)

        assertFalse(outNonCausal.data.zip(outCausal.data.toList()).all { (a, b) -> abs(a - b) < 1e-12 })
    }

    @Test
    fun `backward laeuft ohne exception durch`() {
        val model = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, seed = 42)
        val input = identityInput(ctx = 3, dim = 8)

        val output = model.forward(input, ctx = 3, training = false)
        output.backward()

        model.parameters().forEach { param ->
            assertFalse(param.grad.all { it == 0.0 })
        }
    }

    @Test
    fun `parameters liefert alle vier Gewichtsmatrizen`() {
        val model = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, seed = 1)
        assertEquals(4, model.parameters().size)
    }

    @Test
    fun `forward wirft exception bei falschem input size`() {
        val model = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, seed = 1)
        val wrongInput = Tensor(DoubleArray(10))

        val ex =
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException::class.java) {
                model.forward(wrongInput, ctx = 3, training = false)
            }

        assertEquals("input.size 10 passt nicht zu ctx*embeddingDim=24", ex.message)
    }

    private fun identityInput(
        ctx: Int,
        dim: Int,
    ): Tensor {
        val data =
            DoubleArray(ctx * dim) { idx ->
                val row = idx / dim
                val col = idx % dim
                if (row == col) 1.0 else 0.0
            }
        return Tensor(data)
    }
}
