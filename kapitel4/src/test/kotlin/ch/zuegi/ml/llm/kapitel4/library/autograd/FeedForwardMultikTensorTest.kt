package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class FeedForwardMultikTensorTest {
    @Test
    fun `forward liefert output gleicher laenge wie input`() {
        val ff = FeedForwardMultikTensor(embeddingDim = 4, hiddenDim = 8, seed = 42)
        val input = TensorMultik(mk.ndarray(doubleArrayOf(1.0, 0.0, 0.5, -1.0)))

        val out = ff.forward(input)

        assertEquals(4, out.size)
    }

    @Test
    fun `backward setzt gradienten auf parametern`() {
        val ff = FeedForwardMultikTensor(embeddingDim = 4, hiddenDim = 8, seed = 42)
        val input = TensorMultik(mk.ndarray(doubleArrayOf(0.2, 0.4, 0.6, 0.8)))

        val out = ff.forward(input)
        out.backward()

        val hasGradient =
            ff.parameters().any { p ->
                (0 until p.size).any { i -> p.grad[i] != 0.0 }
            }
        assertEquals(true, hasGradient)
    }

    @Test
    fun `gleiches seed erzeugt gleiche outputs`() {
        val ff1 = FeedForwardMultikTensor(embeddingDim = 4, hiddenDim = 8, seed = 7)
        val ff2 = FeedForwardMultikTensor(embeddingDim = 4, hiddenDim = 8, seed = 7)

        val input = TensorMultik(mk.ndarray(doubleArrayOf(0.1, 0.2, 0.3, 0.4)))

        val out1 = ff1.forward(input)
        val out2 = ff2.forward(input)

        for (i in 0 until out1.size) {
            assertEquals(out1.data[i], out2.data[i], 1e-12)
        }
    }

    @Test
    fun `hiddenDim default ist vierfaches embeddingDim`() {
        val ff = FeedForwardMultikTensor(embeddingDim = 6, seed = 42)

        assertEquals(144, ff.w1.size) // 24 * 6
        assertEquals(24, ff.b1.size)
        assertEquals(144, ff.w2.size) // 6 * 24
        assertEquals(6, ff.b2.size)
    }
}
