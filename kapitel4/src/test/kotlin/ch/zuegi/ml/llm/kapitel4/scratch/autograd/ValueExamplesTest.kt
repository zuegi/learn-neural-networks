package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ValueExamplesTest {
    @Test
    fun `addition gibt gradient unveraendert an beide operanden weiter`() {
        // tag::value-example-plus[]
        val x = Value(2.0)
        val y = Value(3.0)
        val z = x + y

        z.backward()

        assertEquals(5.0, z.data, 1e-12)
        assertEquals(1.0, x.grad, 1e-12)
        assertEquals(1.0, y.grad, 1e-12)
        // end::value-example-plus[]
    }

    @Test
    fun `multiplikation nutzt produktregel`() {
        // tag::value-example-times[]
        val x = Value(2.0)
        val y = Value(3.0)
        val z = x * y

        z.backward()

        assertEquals(6.0, z.data, 1e-12)
        assertEquals(3.0, x.grad, 1e-12)
        assertEquals(2.0, y.grad, 1e-12)
        // end::value-example-times[]
    }

    @Test
    fun `tanh nutzt lokale ableitung aus forward wert`() {
        // tag::value-example-tanh[]
        val x = Value(1.0)
        val y = x.tanh()
        val expectedY = 0.7615941559557649
        val expectedGradient = 1.0 - expectedY * expectedY

        y.backward()

        assertEquals(expectedY, y.data, 1e-12)
        assertEquals(expectedGradient, x.grad, 1e-12)
        // end::value-example-tanh[]
    }

    @Test
    fun `quadrat nutzt potenzregel`() {
        // tag::value-example-pow[]
        val x = Value(3.0)
        val y = x.pow(2.0)

        y.backward()

        assertEquals(9.0, y.data, 1e-12)
        assertEquals(6.0, x.grad, 1e-12)
        // end::value-example-pow[]
    }

    @Test
    fun `division behaelt reihenfolge der operanden`() {
        // tag::value-example-scalar-division[]
        val numerator = Value(4.0)
        val dividedByScalar = numerator / 2.0
        dividedByScalar.backward()

        assertEquals(2.0, dividedByScalar.data, 1e-12)
        assertEquals(0.5, numerator.grad, 1e-12)

        val denominator = Value(4.0)
        val scalarDividedByValue = 2.0 / denominator
        scalarDividedByValue.backward()

        assertEquals(0.5, scalarDividedByValue.data, 1e-12)
        assertEquals(-0.125, denominator.grad, 1e-12)
        // end::value-example-scalar-division[]
    }

    @Test
    fun `vollstaendiges beispiel x mal y plus x`() {
        // tag::value-example-full[]
        val x = Value(2.0)
        val y = Value(3.0)

        val z1 = x * y
        val z = z1 + x

        z.backward()

        assertEquals(6.0, z1.data, 1e-12)
        assertEquals(8.0, z.data, 1e-12)
        assertEquals(1.0, z1.grad, 1e-12)
        assertEquals(4.0, x.grad, 1e-12)
        assertEquals(2.0, y.grad, 1e-12)
        // end::value-example-full[]
    }
}
