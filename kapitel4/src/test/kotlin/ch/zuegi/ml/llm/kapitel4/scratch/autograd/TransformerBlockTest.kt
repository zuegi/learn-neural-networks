package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Test
import kotlin.math.abs

class TransformerBlockTest {
    @Test
    fun `forward liefert output gleicher Form wie input`() {
        val block = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, seed = 42)
        val input = identityInput(ctx = 3, dim = 8)

        val output = block.forward(input, ctx = 3, training = false)

        assertEquals(3 * 8, output.size)
    }

    @Test
    fun `gleiches seed und eval modus erzeugen gleiche outputs`() {
        val b1 = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, seed = 7)
        val b2 = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, seed = 7)

        val input = identityInput(ctx = 3, dim = 8)

        val out1 = b1.forward(input, ctx = 3, training = false)
        val out2 = b2.forward(input, ctx = 3, training = false)

        out1.data.zip(out2.data.toList()).forEach { (a, b) ->
            assertEquals(a, b, 1e-12)
        }
    }

    @Test
    fun `dropout im training beeinflusst output`() {
        val noDropout = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.0, seed = 11)
        val withDropout = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.5, seed = 11)

        val input = identityInput(ctx = 3, dim = 8)

        val out1 = noDropout.forward(input, ctx = 3, training = true)
        val out2 = withDropout.forward(input, ctx = 3, training = true)

        assertFalse(out1.data.zip(out2.data.toList()).all { (a, b) -> abs(a - b) < 1e-12 })
    }

    @Test
    fun `backward laeuft durch und erzeugt nicht-null Gradienten`() {
        val block = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, seed = 42)
        val input = identityInput(ctx = 3, dim = 8)

        val output = block.forward(input, ctx = 3, training = false)
        output.backward()

        assertFalse(block.parameters().all { param -> param.grad.all { it == 0.0 } })
    }

    @Test
    fun `parameters liefert alle lernbaren Gewichte`() {
        val block = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, seed = 1)
        // attentionNorm(2) + attention(4) + feedForwardNorm(2) + feedForward(4)
        assertEquals(12, block.parameters().size)
    }

    @Test
    fun `causal und non-causal liefern unterschiedliche outputs`() {
        val nonCausal = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, causal = false, seed = 99)
        val causal = TransformerBlock(embeddingDim = 8, numHeads = 2, dK = 4, causal = true, seed = 99)

        val input = identityInput(ctx = 3, dim = 8)

        val outNonCausal = nonCausal.forward(input, ctx = 3, training = false)
        val outCausal = causal.forward(input, ctx = 3, training = false)

        assertFalse(outNonCausal.data.zip(outCausal.data.toList()).all { (a, b) -> abs(a - b) < 1e-12 })
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
