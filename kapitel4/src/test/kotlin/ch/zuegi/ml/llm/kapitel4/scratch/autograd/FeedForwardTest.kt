package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test
import kotlin.math.abs

class FeedForwardTest {
    @Test
    fun `forward liefert output gleicher Laenge wie input Zeile`() {
        val ff = FeedForward(embeddingDim = 4, hiddenDim = 8, seed = 42)
        val input = Tensor(doubleArrayOf(1.0, 0.0, 0.5, -1.0))

        val out = ff.forward(input)

        assertEquals(4, out.size)
    }

    @Test
    fun `gleiches seed erzeugt gleiche Outputs`() {
        val ff1 = FeedForward(embeddingDim = 4, hiddenDim = 8, seed = 7)
        val ff2 = FeedForward(embeddingDim = 4, hiddenDim = 8, seed = 7)

        val input = Tensor(doubleArrayOf(0.1, 0.2, 0.3, 0.4))

        val out1 = ff1.forward(input)
        val out2 = ff2.forward(input)

        for (i in out1.data.indices) {
            assertEquals(out1.data[i], out2.data[i], 1e-12)
        }
    }

    @Test
    fun `anderes seed erzeugt andere Outputs`() {
        val ff1 = FeedForward(embeddingDim = 4, hiddenDim = 8, seed = 1)
        val ff2 = FeedForward(embeddingDim = 4, hiddenDim = 8, seed = 2)

        val input = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        val out1 = ff1.forward(input)
        val out2 = ff2.forward(input)

        val allEqual = out1.data.indices.all { i -> abs(out1.data[i] - out2.data[i]) < 1e-12 }
        assertNotEquals(true, allEqual)
    }

    @Test
    fun `hiddenDim default ist vierfaches embeddingDim`() {
        val ff = FeedForward(embeddingDim = 6, seed = 42)

        // w1: [hiddenDim, embeddingDim] flach -> 24 * 6
        assertEquals(24 * 6, ff.w1.size)
        assertEquals(24, ff.b1.size)

        // w2: [embeddingDim, hiddenDim] flach -> 6 * 24
        assertEquals(6 * 24, ff.w2.size)
        assertEquals(6, ff.b2.size)
    }

    @Test
    fun `backward setzt gradients auf Parametern`() {
        val ff = FeedForward(embeddingDim = 4, hiddenDim = 8, seed = 11)
        val input = Tensor(doubleArrayOf(0.2, 0.4, 0.6, 0.8))

        val out = ff.forward(input)
        out.backward()

        val hasGradient = ff.parameters().any { p -> p.grad.any { it != 0.0 } }
        assertEquals(true, hasGradient)
    }
}
