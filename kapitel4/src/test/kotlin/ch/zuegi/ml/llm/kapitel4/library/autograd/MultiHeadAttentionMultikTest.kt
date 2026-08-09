package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.test.DefaultAsserter.assertEquals

class MultiHeadAttentionMultikTest {
    @Test
    fun `forward liefert output mit shape ctx mal embeddingDim`() {
        val mha = MultiHeadAttentionMultik(embeddingDim = 8, numHeads = 2, dK = 4, seed = 42)
        val input = identityInput(ctx = 3, dim = 8)

        val out = mha.forward(input, training = false)

        assertEquals(3, out.shape[0])
        assertEquals(8, out.shape[1])
    }

    @Test
    fun `gleiches seed und eval modus erzeugen gleiche outputs`() {
        val m1 = MultiHeadAttentionMultik(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.3, seed = 7)
        val m2 = MultiHeadAttentionMultik(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.3, seed = 7)

        val input = identityInput(ctx = 3, dim = 8)

        val out1 = m1.forward(input, training = false)
        val out2 = m2.forward(input, training = false)

        for (r in 0 until out1.shape[0]) {
            for (c in 0 until out1.shape[1]) {
                assertEquals(out1[r][c], out2[r][c], 1e-12)
            }
        }
    }

    @Test
    fun `causal und non-causal liefern unterschiedliche outputs`() {
        val nonCausal = MultiHeadAttentionMultik(embeddingDim = 8, numHeads = 2, dK = 4, causal = false, seed = 99)
        val causal = MultiHeadAttentionMultik(embeddingDim = 8, numHeads = 2, dK = 4, causal = true, seed = 99)
        val input = identityInput(ctx = 3, dim = 8)

        val out1 = nonCausal.forward(input, training = false)
        val out2 = causal.forward(input, training = false)

        val allEqual =
            (0 until out1.shape[0]).all { r ->
                (0 until out1.shape[1]).all { c -> abs(out1[r][c] - out2[r][c]) < 1e-12 }
            }

        assertFalse(allEqual)
    }

    @Test
    fun `dropout im training beeinflusst output`() {
        val noDropout = MultiHeadAttentionMultik(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.0, seed = 11)
        val withDropout = MultiHeadAttentionMultik(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.5, seed = 11)
        val input = identityInput(ctx = 3, dim = 8)

        val out1 = noDropout.forward(input, training = true)
        val out2 = withDropout.forward(input, training = true)

        val allEqual =
            (0 until out1.shape[0]).all { r ->
                (0 until out1.shape[1]).all { c -> abs(out1[r][c] - out2[r][c]) < 1e-12 }
            }

        assertFalse(allEqual)
    }

    private fun identityInput(
        ctx: Int,
        dim: Int,
    ) = mk.ndarray(
        List(ctx) { row ->
            List(dim) { col -> if (row == col) 1.0 else 0.0 }
        },
    )
}
