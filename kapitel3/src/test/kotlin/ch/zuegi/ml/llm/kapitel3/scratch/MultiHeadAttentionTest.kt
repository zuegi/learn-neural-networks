package ch.zuegi.ml.llm.kapitel3.scratch

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test
import kotlin.math.abs

class MultiHeadAttentionTest {
    @Test
    fun `forward liefert output mit shape context x embeddingDim`() {
        val model = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, seed = 42)
        val input =
            matrixOf(
                doubleArrayOf(1.0, 0.5, -1.0, 0.2, 0.3, 0.1, -0.5, 0.8),
                doubleArrayOf(0.2, -0.1, 0.3, 0.4, 0.6, 0.7, 0.0, -0.3),
                doubleArrayOf(-0.3, 0.7, 0.0, 0.1, 0.5, -0.2, 0.4, 0.9),
            )

        val output = model.forward(input, training = false)

        assertEquals(3, output.size)
        assertEquals(8, output[0].size)
    }

    @Test
    fun `gleiches seed und eval modus erzeugen gleiche outputs`() {
        val m1 = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.3, seed = 99)
        val m2 = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.3, seed = 99)

        val input =
            matrixOf(
                doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                doubleArrayOf(0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2),
                doubleArrayOf(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2),
            )

        val out1 = m1.forward(input, training = false)
        val out2 = m2.forward(input, training = false)

        assertMatrixEquals(out1, out2, 1e-12)
    }

    @Test
    fun `dropout im training beeinflusst output`() {
        val noDropout = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.0, seed = 11)
        val withDropout = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, dropoutProb = 0.5, seed = 11)

        val input =
            matrixOf(
                doubleArrayOf(0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7),
                doubleArrayOf(0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2),
            )

        val outNoDropout = noDropout.forward(input, training = true)
        val outWithDropout = withDropout.forward(input, training = true)

        assertNotEquals(true, matrixAlmostEqual(outNoDropout, outWithDropout, eps = 1e-12))
    }

    @Test
    fun `forward wirft exception bei falscher input embedding dim`() {
        val model = MultiHeadAttention(embeddingDim = 8, numHeads = 2, dK = 4, seed = 1)
        val wrongInput =
            matrixOf(
                doubleArrayOf(1.0, 2.0, 3.0, 4.0), // 4 statt 8
            )

        val ex =
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException::class.java) {
                model.forward(wrongInput, training = false)
            }

        assertEquals("Input-Embedding-Dim 4 passt nicht zu embeddingDim 8", ex.message)
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
