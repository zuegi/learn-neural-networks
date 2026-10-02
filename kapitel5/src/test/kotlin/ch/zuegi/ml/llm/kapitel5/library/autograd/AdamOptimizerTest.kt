package ch.zuegi.ml.llm.kapitel5.library.autograd

import ch.zuegi.ml.llm.kapitel4.library.autograd.TensorMultik
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class AdamOptimizerTest {
    @Test
    fun `erster Schritt bewegt jeden Parameter um etwa learningRate gegen das Gradientenvorzeichen`() {
        val parameter = TensorMultik(mk.ndarray(doubleArrayOf(1.0, 1.0, 1.0)))
        val gradients = doubleArrayOf(4.0, -0.001, 250.0)
        gradients.forEachIndexed { i, g -> parameter.grad[i] = g }
        val optimizer = AdamOptimizer(listOf(parameter), learningRate = LEARNING_RATE)

        optimizer.step()

        assertEquals(1.0 - LEARNING_RATE, parameter.data[0], TOLERANCE)
        assertEquals(1.0 + LEARNING_RATE, parameter.data[1], TOLERANCE)
        assertEquals(1.0 - LEARNING_RATE, parameter.data[2], TOLERANCE)
    }

    @Test
    fun `zeroGrad setzt alle Gradienten auf null`() {
        val parameter = TensorMultik(mk.ndarray(doubleArrayOf(1.0, 2.0)))
        parameter.grad[0] = 3.0
        parameter.grad[1] = -3.0

        AdamOptimizer(listOf(parameter)).zeroGrad()

        assertEquals(0.0, parameter.grad[0])
        assertEquals(0.0, parameter.grad[1])
    }

    companion object {
        private const val LEARNING_RATE = 0.01
        private const val TOLERANCE = 1e-6
    }
}
