package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import kotlin.math.abs

class LayerNormMultikTest {
    @Test
    fun `forward liefert gleiche Laenge wie input`() {
        val ln = LayerNormMultik(embeddingDim = 4)
        val x = mk.ndarray(doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        val out = ln.forward(x)

        assertEquals(4, out.size)
    }

    @Test
    fun `default gamma und beta sind korrekt initialisiert`() {
        val ln = LayerNormMultik(embeddingDim = 5)

        for (i in 0 until 5) {
            assertEquals(1.0, ln.gamma[i], 1e-12)
            assertEquals(0.0, ln.beta[i], 1e-12)
        }
    }

    @Test
    fun `deterministisch bei gleichem input`() {
        val ln = LayerNormMultik(embeddingDim = 4)
        val x = mk.ndarray(doubleArrayOf(0.5, -1.0, 2.0, 1.5))

        val out1 = ln.forward(x)
        val out2 = ln.forward(x)

        for (i in 0 until out1.size) {
            assertEquals(out1[i], out2[i], 1e-12)
        }
    }

    @Test
    fun `output hat bei default gamma beta ungefaehr mean null`() {
        val ln = LayerNormMultik(embeddingDim = 4)
        val x = mk.ndarray(doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        val out = ln.forward(x)
        val mean = (0 until out.size).sumOf { out[it] } / out.size

        assertEquals(0.0, mean, 1e-9)
        val hasFinite = (0 until out.size).all { !out[it].isNaN() && !out[it].isInfinite() }
        assertEquals(true, hasFinite)
    }

    @Test
    fun `constructor wirft exception bei embeddingDim null oder negativ`() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException::class.java) {
            LayerNormMultik(embeddingDim = 0)
        }
    }
}
