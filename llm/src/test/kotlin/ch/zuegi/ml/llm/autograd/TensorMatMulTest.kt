package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test

class TensorMatMulTest {
    @Test
    fun `matMul computes correct result`() {
        // A = [[1,2],[3,4]] (2x2), B = [[5,6],[7,8]] (2x2)
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val b = Tensor(doubleArrayOf(5.0, 6.0, 7.0, 8.0))

        val c = a.matMul(b, p = 2, q = 2, r = 2)

        // C = [[19,22],[43,50]]
        assertThat(c.data).containsExactly(19.0, 22.0, 43.0, 50.0)
    }

    @Test
    fun `matMul non-square computes correct result`() {
        // A = [[1,2,3]] (1x3), B = [[1],[0],[-1]] (3x1) -> C = [[-2]] (1x1)
        val a = Tensor(doubleArrayOf(1.0, 2.0, 3.0))
        val b = Tensor(doubleArrayOf(1.0, 0.0, -1.0))

        val c = a.matMul(b, p = 1, q = 3, r = 1)

        assertThat(c.data).containsExactly(-2.0)
    }

    @Test
    fun `matMul left gradient matches numeric`() {
        val aVal = doubleArrayOf(0.5, -0.3, 0.8, 1.1, -0.6, 0.2) // 2x3
        val bVal = doubleArrayOf(0.9, -0.4, 0.7, 1.2, -0.1, 0.3) // 3x2

        fun forwardSum(a: DoubleArray): Double {
            val p = 2
            val q = 3
            val r = 2
            var total = 0.0
            for (i in 0 until p) {
                for (j in 0 until r) {
                    var sum = 0.0
                    for (k in 0 until q) sum += a[i * q + k] * bVal[k * r + j]
                    total += sum
                }
            }
            return total
        }

        val a = Tensor(aVal)
        val b = Tensor(bVal)
        val c = a.matMul(b, p = 2, q = 3, r = 2)
        c.backward()

        val h = 1e-6
        for (i in aVal.indices) {
            val plus = aVal.copyOf().also { it[i] += h }
            val minus = aVal.copyOf().also { it[i] -= h }
            val numeric = (forwardSum(plus) - forwardSum(minus)) / (2 * h)
            assertThat(a.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }

    @Test
    fun `matMul right gradient matches numeric`() {
        val aVal = doubleArrayOf(0.5, -0.3, 0.8, 1.1, -0.6, 0.2) // 2x3
        val bVal = doubleArrayOf(0.9, -0.4, 0.7, 1.2, -0.1, 0.3) // 3x2

        fun forwardSum(b: DoubleArray): Double {
            val p = 2
            val q = 3
            val r = 2
            var total = 0.0
            for (i in 0 until p) {
                for (j in 0 until r) {
                    var sum = 0.0
                    for (k in 0 until q) sum += aVal[i * q + k] * b[k * r + j]
                    total += sum
                }
            }
            return total
        }

        val a = Tensor(aVal)
        val b = Tensor(bVal)
        val c = a.matMul(b, p = 2, q = 3, r = 2)
        c.backward()

        val h = 1e-6
        for (i in bVal.indices) {
            val plus = bVal.copyOf().also { it[i] += h }
            val minus = bVal.copyOf().also { it[i] -= h }
            val numeric = (forwardSum(plus) - forwardSum(minus)) / (2 * h)
            assertThat(b.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }

    @Test
    fun `matMul chained with tanh matches numeric`() {
        val aVal = doubleArrayOf(0.3, -0.5, 0.8, 0.1) // 2x2
        val bVal = doubleArrayOf(0.6, -0.2, 0.4, 0.9) // 2x2

        fun forwardSum(a: DoubleArray): Double {
            val p = 2
            val q = 2
            val r = 2
            var total = 0.0
            for (i in 0 until p) {
                for (j in 0 until r) {
                    var sum = 0.0
                    for (k in 0 until q) sum += a[i * q + k] * bVal[k * r + j]
                    total += kotlin.math.tanh(sum)
                }
            }
            return total
        }

        val a = Tensor(aVal)
        val b = Tensor(bVal)
        val c = a.matMul(b, p = 2, q = 2, r = 2).tanh()
        c.backward()

        val h = 1e-6
        for (i in aVal.indices) {
            val plus = aVal.copyOf().also { it[i] += h }
            val minus = aVal.copyOf().also { it[i] -= h }
            val numeric = (forwardSum(plus) - forwardSum(minus)) / (2 * h)
            assertThat(a.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }
}
