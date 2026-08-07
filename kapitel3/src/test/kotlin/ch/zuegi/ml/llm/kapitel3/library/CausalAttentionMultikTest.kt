package ch.zuegi.ml.llm.kapitel3.ch.zuegi.ml.llm.kapitel3.library

import ch.zuegi.ml.llm.kapitel3.library.CausalAttentionMultik
import ch.zuegi.ml.llm.kapitel3.library.SelfAttentionMultik
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import kotlin.math.abs

class CausalAttentionMultikTest {
    @Test
    fun `forward liefert output mit shape context x dK`() {
        val model =
            CausalAttentionMultik(
                embeddingDim = 4,
                dK = 3,
                dropoutProb = 0.0,
                seed = 42,
            )
        val input =
            matrixOf(
                listOf(
                    listOf(1.0, 0.0, 0.5, -1.0),
                    listOf(0.2, -0.1, 0.3, 0.4),
                    listOf(-0.3, 0.7, 0.0, 0.1),
                ),
            )

        val output = model.forward(input, training = false)

        assertEquals(3, output.shape[0])
        assertEquals(3, output.shape[1])
    }

    @Test
    fun `gleiches seed und eval modus erzeugen gleiche outputs`() {
        val m1 =
            CausalAttentionMultik(
                embeddingDim = 4,
                dK = 3,
                dropoutProb = 0.3,
                seed = 7,
            )
        val m2 =
            CausalAttentionMultik(
                embeddingDim = 4,
                dK = 3,
                dropoutProb = 0.3,
                seed = 7,
            )

        val input =
            matrixOf(
                listOf(
                    listOf(0.1, 0.2, 0.3, 0.4),
                    listOf(0.5, 0.6, 0.7, 0.8),
                    listOf(0.9, 1.0, 1.1, 1.2),
                ),
            )

        val out1 = m1.forward(input, training = false)
        val out2 = m2.forward(input, training = false)

        assertMatrixEquals(out1, out2, 1e-12)
    }

    @Test
    fun `causal attention unterscheidet sich von self attention ohne maskierung`() {
        val causal =
            CausalAttentionMultik(
                embeddingDim = 4,
                dK = 3,
                dropoutProb = 0.0,
                seed = 99,
            )
        val nonCausal =
            SelfAttentionMultik(
                embeddingDim = 4,
                dK = 3,
                causal = false,
                seed = 99,
            )

        val input =
            matrixOf(
                listOf(
                    listOf(1.0, 2.0, 3.0, 4.0),
                    listOf(4.0, 3.0, 2.0, 1.0),
                    listOf(1.5, 1.0, 0.5, 0.0),
                ),
            )

        val outCausal = causal.forward(input, training = false)
        val outNonCausal = nonCausal.forward(input)

        val same = matrixAlmostEqual(outCausal, outNonCausal, eps = 1e-12)
        assertFalse(same)
    }

    @Test
    fun `dropout im training beeinflusst output`() {
        val noDropout =
            CausalAttentionMultik(
                embeddingDim = 4,
                dK = 3,
                dropoutProb = 0.0,
                seed = 11,
            )
        val withDropout =
            CausalAttentionMultik(
                embeddingDim = 4,
                dK = 3,
                dropoutProb = 0.5,
                seed = 11,
            )

        val input =
            matrixOf(
                listOf(
                    listOf(0.2, 0.4, 0.6, 0.8),
                    listOf(0.1, 0.3, 0.5, 0.7),
                    listOf(0.9, 0.7, 0.5, 0.3),
                ),
            )

        val outNoDropout = noDropout.forward(input, training = true)
        val outWithDropout = withDropout.forward(input, training = true)

        val same = matrixAlmostEqual(outNoDropout, outWithDropout, eps = 1e-12)
        assertNotEquals(true, same)
    }

    @Test
    fun `forward wirft exception bei falscher input embedding dim`() {
        val model =
            CausalAttentionMultik(
                embeddingDim = 4,
                dK = 2,
                dropoutProb = 0.0,
                seed = 1,
            )
        val wrongInput =
            matrixOf(
                listOf(
                    listOf(1.0, 2.0, 3.0),
                    listOf(4.0, 5.0, 6.0),
                ),
            )

        val ex =
            assertThrows(IllegalArgumentException::class.java) {
                model.forward(wrongInput, training = false)
            }

        assertEquals(
            "Input-Embedding-Dim 3 passt nicht zu embeddingDim 4",
            ex.message,
        )
    }

    private fun matrixOf(rows: List<List<Double>>): NDArray<Double, D2> = mk.ndarray(rows)

    private fun assertMatrixEquals(
        actual: NDArray<Double, D2>,
        expected: NDArray<Double, D2>,
        eps: Double,
    ) {
        assertEquals(expected.shape[0], actual.shape[0])
        assertEquals(expected.shape[1], actual.shape[1])

        for (i in 0 until actual.shape[0]) {
            val aRow = DoubleArray(actual.shape[1]) { j -> actual[i][j] }
            val eRow = DoubleArray(expected.shape[1]) { j -> expected[i][j] }
            assertArrayEquals(eRow, aRow, eps)
        }
    }

    private fun matrixAlmostEqual(
        a: NDArray<Double, D2>,
        b: NDArray<Double, D2>,
        eps: Double,
    ): Boolean {
        if (a.shape[0] != b.shape[0] || a.shape[1] != b.shape[1]) return false

        for (i in 0 until a.shape[0]) {
            for (j in 0 until a.shape[1]) {
                if (abs(a[i][j] - b[i][j]) > eps) return false
            }
        }
        return true
    }
}
