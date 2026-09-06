package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TensorExamplesTest {
    @Test
    fun `vektorielle addition reicht gradient elementweise weiter`() {
        // tag::tensor-example-plus[]
        val x = Tensor(doubleArrayOf(1.0, 2.0, 3.0))
        val y = Tensor(doubleArrayOf(2.0, 3.0, 4.0))
        val z = x + y

        z.backward()

        assertEquals(3, z.size)
        assertArrayEquals(doubleArrayOf(3.0, 5.0, 7.0), z.data, 1e-12)
        assertArrayEquals(doubleArrayOf(1.0, 1.0, 1.0), x.grad, 1e-12)
        assertArrayEquals(doubleArrayOf(1.0, 1.0, 1.0), y.grad, 1e-12)
        // end::tensor-example-plus[]
    }

    @Test
    fun `matrixmultiplikation berechnet werte und gradienten`() {
        // tag::tensor-example-matmul[]
        val p = 2
        val q = 2
        val r = 2
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val b = Tensor(doubleArrayOf(5.0, 6.0, 7.0, 8.0))
        val c = a.matMul(b, p, q, r)

        c.backward()

        assertEquals(p * q, a.size)
        assertEquals(q * r, b.size)
        assertEquals(p * r, c.size)
        assertArrayEquals(doubleArrayOf(19.0, 22.0, 43.0, 50.0), c.data, 1e-12)
        assertArrayEquals(doubleArrayOf(11.0, 15.0, 11.0, 15.0), a.grad, 1e-12)
        assertArrayEquals(doubleArrayOf(4.0, 4.0, 6.0, 6.0), b.grad, 1e-12)
        // end::tensor-example-matmul[]
    }

    @Test
    fun `zeroGrad setzt akkumulierten gradienten zurueck`() {
        // tag::tensor-example-zero-grad[]
        val parameter = Tensor(doubleArrayOf(3.0))
        val loss = parameter * parameter
        loss.backward()
        assertArrayEquals(doubleArrayOf(6.0), parameter.grad, 1e-12)

        parameter.zeroGrad()

        assertArrayEquals(doubleArrayOf(0.0), parameter.grad, 1e-12)
        // end::tensor-example-zero-grad[]
    }
}
