package ch.zuegi.ml.llm.kapitel4.library.multik

import ch.zuegi.ml.llm.kapitel4.library.autograd.FeedForwardMultik
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import kotlin.math.abs

class FeedForwardMultikTest {
    @Test
    fun `forward liefert output gleicher Laenge wie input`() {
        val ff = FeedForwardMultik(embeddingDim = 4, hiddenDim = 8, seed = 42)
        val input = mk.ndarray(doubleArrayOf(1.0, 0.0, 0.5, -1.0))

        val out = ff.forward(input)

        assertEquals(4, out.size)
    }

    @Test
    fun `gleiches seed erzeugt gleiche Outputs`() {
        val ff1 = FeedForwardMultik(embeddingDim = 4, hiddenDim = 8, seed = 7)
        val ff2 = FeedForwardMultik(embeddingDim = 4, hiddenDim = 8, seed = 7)

        val input = mk.ndarray(doubleArrayOf(0.1, 0.2, 0.3, 0.4))

        val out1 = ff1.forward(input)
        val out2 = ff2.forward(input)

        for (i in 0 until out1.size) {
            assertEquals(out1[i], out2[i], 1e-12)
        }
    }

    @Test
    fun `anderes seed erzeugt andere Outputs`() {
        val ff1 = FeedForwardMultik(embeddingDim = 4, hiddenDim = 8, seed = 1)
        val ff2 = FeedForwardMultik(embeddingDim = 4, hiddenDim = 8, seed = 2)

        val input = mk.ndarray(doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        val out1 = ff1.forward(input)
        val out2 = ff2.forward(input)

        val allEqual = (0 until out1.size).all { i -> abs(out1[i] - out2[i]) < 1e-12 }
        assertEquals(false, allEqual)
    }

    @Test
    fun `hiddenDim default ist vierfaches embeddingDim`() {
        val ff = FeedForwardMultik(embeddingDim = 6, seed = 42)

        assertEquals(24, ff.w1.shape[0]) // hiddenDim
        assertEquals(6, ff.w1.shape[1]) // embeddingDim
        assertEquals(24, ff.b1.size)

        assertEquals(6, ff.w2.shape[0]) // embeddingDim
        assertEquals(24, ff.w2.shape[1]) // hiddenDim
        assertEquals(6, ff.b2.size)
    }

    @Test
    fun `forward wirft exception bei falscher input laenge`() {
        val ff = FeedForwardMultik(embeddingDim = 4, hiddenDim = 8, seed = 42)
        val wrong = mk.ndarray(doubleArrayOf(1.0, 2.0, 3.0))

        val ex =
            assertThrows(IllegalArgumentException::class.java) {
                ff.forward(wrong)
            }

        assertEquals("x.size 3 passt nicht zu embeddingDim 4", ex.message)
    }
}
