package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class ValueExamplesTest {
    @Test
    fun `addition verteilt gradient 1 zu 1`() {
        // tag::value-example-plus[]
        val x = Value(2.0)
        val y = Value(3.0)
        val z = x + y

        z.backward()

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

        assertEquals(3.0, x.grad, 1e-12)
        assertEquals(2.0, y.grad, 1e-12)
        // end::value-example-times[]
    }

    @Test
    fun `vollstaendiges beispiel x mal y plus x`() {
        // tag::value-example-full[]
        val x = Value(2.0)
        val y = Value(3.0)

        val z1 = x * y
        val z = z1 + x

        z.backward()

        assertEquals(4.0, x.grad, 1e-12)
        assertEquals(2.0, y.grad, 1e-12)
        // end::value-example-full[]
    }
}
