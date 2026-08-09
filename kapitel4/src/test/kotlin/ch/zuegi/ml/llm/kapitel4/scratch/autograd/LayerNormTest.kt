package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Test
import kotlin.math.abs

class LayerNormTest {
    @Test
    fun `forward liefert gleiche Laenge wie input`() {
        val layerNorm = LayerNorm(embeddingDim = 4)
        val x = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        val out = layerNorm.forward(x)

        assertEquals(4, out.size)
    }

    @Test
    fun `default gamma und beta haben erwartete Initialwerte`() {
        val layerNorm = LayerNorm(embeddingDim = 5)

        assertEquals(5, layerNorm.gamma.size)
        assertEquals(5, layerNorm.beta.size)
        assertFalse(layerNorm.gamma.data.any { abs(it - 1.0) > 1e-12 })
        assertFalse(layerNorm.beta.data.any { abs(it) > 1e-12 })
    }

    @Test
    fun `parameters liefert gamma und beta`() {
        val layerNorm = LayerNorm(embeddingDim = 3)

        val params = layerNorm.parameters()

        assertEquals(2, params.size)
        assertEquals(layerNorm.gamma, params[0])
        assertEquals(layerNorm.beta, params[1])
    }

    @Test
    fun `backward setzt Gradienten fuer input gamma beta`() {
        val layerNorm = LayerNorm(embeddingDim = 4)
        val x = Tensor(doubleArrayOf(0.5, -1.0, 2.0, 1.5))

        val out = layerNorm.forward(x)
        out.backward()

        val hasAnyGrad =
            x.grad.any { it != 0.0 } ||
                layerNorm.gamma.grad.any { it != 0.0 } ||
                layerNorm.beta.grad.any { it != 0.0 }

        assertEquals(true, hasAnyGrad)
    }

    @Test
    fun `deterministisch bei gleichem input und Parametern`() {
        val layerNorm = LayerNorm(embeddingDim = 4)
        val x = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        val out1 = layerNorm.forward(x)
        val out2 = layerNorm.forward(x)

        for (i in out1.data.indices) {
            assertEquals(out1.data[i], out2.data[i], 1e-12)
        }
    }
}
