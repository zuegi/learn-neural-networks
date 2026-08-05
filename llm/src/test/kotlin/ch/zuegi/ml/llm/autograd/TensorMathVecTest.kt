package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test

class TensorMathVecTest {
    @Test
    fun `matVecMul computes correct result`() {
        // W = [[1,2,3],[4,5,6]] (2x3), x = [1,0,-1]
        val x = Tensor(doubleArrayOf(1.0, 0.0, -1.0))
        val w = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

        val y = x.matVecMul(w, m = 2, n = 3)

        // y[0] = 1*1 + 2*0 + 3*(-1) = -2
        // y[1] = 4*1 + 5*0 + 6*(-1) = -2
        assertThat(y.data).containsExactly(-2.0, -2.0)
    }

    @Test
    fun `matVecMul input gradient matches numeric`() {
        val xVal = doubleArrayOf(0.5, -0.3, 0.8)
        val wVal = doubleArrayOf(1.2, -0.7, 0.4, 0.9, 0.1, -1.1) // 2x3

        fun forwardSum(x: DoubleArray): Double {
            val m = 2
            val n = 3
            var total = 0.0
            for (row in 0 until m) {
                var sum = 0.0
                for (col in 0 until n) sum += wVal[row * n + col] * x[col]
                total += sum
            }
            return total
        }

        val x = Tensor(xVal)
        val w = Tensor(wVal)
        val y = x.matVecMul(w, m = 2, n = 3)
        y.backward()

        val h = 1e-6
        for (i in xVal.indices) {
            val plus = xVal.copyOf().also { it[i] += h }
            val minus = xVal.copyOf().also { it[i] -= h }
            val numeric = (forwardSum(plus) - forwardSum(minus)) / (2 * h)
            assertThat(x.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }

    @Test
    fun `matVecMul weight gradient matches numeric`() {
        val xVal = doubleArrayOf(0.5, -0.3, 0.8)
        val wVal = doubleArrayOf(1.2, -0.7, 0.4, 0.9, 0.1, -1.1) // 2x3

        fun forwardSum(w: DoubleArray): Double {
            val m = 2
            val n = 3
            var total = 0.0
            for (row in 0 until m) {
                var sum = 0.0
                for (col in 0 until n) sum += w[row * n + col] * xVal[col]
                total += sum
            }
            return total
        }

        val x = Tensor(xVal)
        val w = Tensor(wVal)
        val y = x.matVecMul(w, m = 2, n = 3)
        y.backward()

        val h = 1e-6
        for (i in wVal.indices) {
            val plus = wVal.copyOf().also { it[i] += h }
            val minus = wVal.copyOf().also { it[i] -= h }
            val numeric = (forwardSum(plus) - forwardSum(minus)) / (2 * h)
            assertThat(w.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }

    @Test
    fun `matVecMul chained with tanh matches numeric`() {
        val xVal = doubleArrayOf(0.3, -0.5)
        val wVal = doubleArrayOf(0.8, -0.2, 0.5, 1.1) // 2x2

        fun forwardSum(x: DoubleArray): Double {
            val m = 2
            val n = 2
            var total = 0.0
            for (row in 0 until m) {
                var sum = 0.0
                for (col in 0 until n) sum += wVal[row * n + col] * x[col]
                total += kotlin.math.tanh(sum)
            }
            return total
        }

        val x = Tensor(xVal)
        val w = Tensor(wVal)
        val y = x.matVecMul(w, m = 2, n = 2).tanh()
        y.backward()

        val h = 1e-6
        for (i in xVal.indices) {
            val plus = xVal.copyOf().also { it[i] += h }
            val minus = xVal.copyOf().also { it[i] -= h }
            val numeric = (forwardSum(plus) - forwardSum(minus)) / (2 * h)
            assertThat(x.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }
}
