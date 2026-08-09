package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TensorMultikTest {
    @Test
    fun `plus setzt gradienten korrekt`() {
        val x = TensorMultik(mk.ndarray(doubleArrayOf(1.0, 2.0)))
        val y = TensorMultik(mk.ndarray(doubleArrayOf(3.0, 4.0)))
        val z = x + y

        z.backward()

        assertEquals(1.0, x.grad[0], 1e-12)
        assertEquals(1.0, x.grad[1], 1e-12)
        assertEquals(1.0, y.grad[0], 1e-12)
        assertEquals(1.0, y.grad[1], 1e-12)
    }

    @Test
    fun `times multipliziert korrekt`() {
        val x = TensorMultik(mk.ndarray(doubleArrayOf(2.0, 3.0)))
        val y = TensorMultik(mk.ndarray(doubleArrayOf(4.0, 5.0)))
        val z = x * y

        assertEquals(8.0, z.data[0], 1e-12)
        assertEquals(15.0, z.data[1], 1e-12)
    }

    @Test
    fun `dot product korrekt`() {
        val x = TensorMultik(mk.ndarray(doubleArrayOf(1.0, 2.0, 3.0)))
        val y = TensorMultik(mk.ndarray(doubleArrayOf(4.0, 5.0, 6.0)))
        val z = x.dot(y)

        assertEquals(32.0, z.data[0], 1e-12) // 1*4 + 2*5 + 3*6
    }
}
