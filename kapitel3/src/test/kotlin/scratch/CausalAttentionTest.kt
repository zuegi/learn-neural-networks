package ch.zuegi.ml.llm.kapitel3.scratch

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test
import kotlin.math.abs

class CausalAttentionTest {
    @Test
    fun `forward liefert output mit shape context x dK`() {
        val model = CausalAttention(embeddingDim = 4, dK = 3, dropoutProb = 0.0, seed = 42)
        val input =
            matrixOf(
                doubleArrayOf(1.0, 0.0, 0.5, -1.0),
                doubleArrayOf(0.2, -0.1, 0.3, 0.4),
                doubleArrayOf(-0.3, 0.7, 0.0, 0.1),
            )

        val output = model.forward(input, training = false)

        assertEquals(3, output.size)
        assertEquals(3, output[0].size)
    }

    @Test
    fun `gleiches seed und eval modus erzeugen gleiche outputs`() {
        val m1 = CausalAttention(embeddingDim = 4, dK = 3, dropoutProb = 0.3, seed = 7)
        val m2 = CausalAttention(embeddingDim = 4, dK = 3, dropoutProb = 0.3, seed = 7)

        val input =
            matrixOf(
                doubleArrayOf(0.1, 0.2, 0.3, 0.4),
                doubleArrayOf(0.5, 0.6, 0.7, 0.8),
                doubleArrayOf(0.9, 1.0, 1.1, 1.2),
            )

        val out1 = m1.forward(input, training = false)
        val out2 = m2.forward(input, training = false)

        assertMatrixEquals(out1, out2, 1e-12)
    }

    @Test
    fun `causal attention unterscheidet sich von self attention ohne maskierung`() {
        val causal = CausalAttention(embeddingDim = 4, dK = 3, dropoutProb = 0.0, seed = 99)
        val nonCausal = SelfAttention(embeddingDim = 4, dK = 3, causal = false, seed = 99)

        val input =
            matrixOf(
                doubleArrayOf(1.0, 2.0, 3.0, 4.0),
                doubleArrayOf(4.0, 3.0, 2.0, 1.0),
                doubleArrayOf(1.5, 1.0, 0.5, 0.0),
            )

        val outCausal = causal.forward(input, training = false)
        val outNonCausal = nonCausal.forward(input)

        val same = matrixAlmostEqual(outCausal, outNonCausal, eps = 1e-12)
        assertFalse(same)
    }

    @Test
    fun `dropout im training beeinflusst output`() {
        val noDropout = CausalAttention(embeddingDim = 4, dK = 3, dropoutProb = 0.0, seed = 11)
        val withDropout = CausalAttention(embeddingDim = 4, dK = 3, dropoutProb = 0.5, seed = 11)

        val input =
            matrixOf(
                doubleArrayOf(0.2, 0.4, 0.6, 0.8),
                doubleArrayOf(0.1, 0.3, 0.5, 0.7),
                doubleArrayOf(0.9, 0.7, 0.5, 0.3),
            )

        val outNoDropout = noDropout.forward(input, training = true)
        val outWithDropout = withDropout.forward(input, training = true)

        val same = matrixAlmostEqual(outNoDropout, outWithDropout, eps = 1e-12)
        assertNotEquals(true, same)
    }

    private fun matrixOf(vararg rows: DoubleArray): Array<DoubleArray> = arrayOf(*rows)

    private fun assertMatrixEquals(
        actual: Array<DoubleArray>,
        expected: Array<DoubleArray>,
        eps: Double,
    ) {
        assertEquals(expected.size, actual.size)
        assertEquals(expected[0].size, actual[0].size)

        for (i in actual.indices) {
            assertArrayEquals(expected[i], actual[i], eps)
        }
    }

    private fun matrixAlmostEqual(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
        eps: Double,
    ): Boolean {
        if (a.size != b.size || a[0].size != b[0].size) return false

        for (i in a.indices) {
            for (j in a[i].indices) {
                if (abs(a[i][j] - b[i][j]) > eps) return false
            }
        }
        return true
    }
}
