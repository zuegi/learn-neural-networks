package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.sqrt

class TensorLayerNormTest {
    private fun layerNormSum(
        x: DoubleArray,
        gamma: DoubleArray,
        beta: DoubleArray,
        eps: Double = 1e-5,
    ): Double {
        val n = x.size
        val mean = x.average()
        val variance = x.sumOf { (it - mean) * (it - mean) } / n
        val std = sqrt(variance + eps)
        var total = 0.0
        for (i in 0 until n) {
            total += gamma[i] * ((x[i] - mean) / std) + beta[i]
        }
        return total
    }

    @Test
    fun `output has mean approx zero and variance approx one with default params`() {
        val x = Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val gamma = Tensor(DoubleArray(4) { 1.0 })
        val beta = Tensor(DoubleArray(4) { 0.0 })

        val y = x.layerNorm(gamma, beta)

        val mean = y.data.average()
        val variance = y.data.sumOf { (it - mean) * (it - mean) } / y.size
        assertThat(mean).isCloseTo(0.0, offset(1e-9))
        assertThat(variance).isCloseTo(1.0, offset(1e-4))
    }

    @Test
    fun `input gradient matches numeric`() {
        val xVal = doubleArrayOf(0.5, -1.2, 0.8, 2.1)
        val gammaVal = doubleArrayOf(1.1, 0.9, 1.3, 0.7)
        val betaVal = doubleArrayOf(0.2, -0.1, 0.05, 0.4)

        val x = Tensor(xVal)
        val gamma = Tensor(gammaVal)
        val beta = Tensor(betaVal)
        x.layerNorm(gamma, beta).backward()

        val h = 1e-6
        for (i in xVal.indices) {
            val plus = xVal.copyOf().also { it[i] += h }
            val minus = xVal.copyOf().also { it[i] -= h }
            val numeric = (layerNormSum(plus, gammaVal, betaVal) - layerNormSum(minus, gammaVal, betaVal)) / (2 * h)
            assertThat(x.grad[i]).isCloseTo(numeric, offset(1e-5))
        }
    }

    @Test
    fun `gamma gradient matches numeric`() {
        val xVal = doubleArrayOf(0.5, -1.2, 0.8, 2.1)
        val gammaVal = doubleArrayOf(1.1, 0.9, 1.3, 0.7)
        val betaVal = doubleArrayOf(0.2, -0.1, 0.05, 0.4)

        val x = Tensor(xVal)
        val gamma = Tensor(gammaVal)
        val beta = Tensor(betaVal)
        x.layerNorm(gamma, beta).backward()

        val h = 1e-6
        for (i in gammaVal.indices) {
            val plus = gammaVal.copyOf().also { it[i] += h }
            val minus = gammaVal.copyOf().also { it[i] -= h }
            val numeric = (layerNormSum(xVal, plus, betaVal) - layerNormSum(xVal, minus, betaVal)) / (2 * h)
            assertThat(gamma.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }

    @Test
    fun `beta gradient matches numeric`() {
        val xVal = doubleArrayOf(0.5, -1.2, 0.8, 2.1)
        val gammaVal = doubleArrayOf(1.1, 0.9, 1.3, 0.7)
        val betaVal = doubleArrayOf(0.2, -0.1, 0.05, 0.4)

        val x = Tensor(xVal)
        val gamma = Tensor(gammaVal)
        val beta = Tensor(betaVal)
        x.layerNorm(gamma, beta).backward()

        // dbeta = dy = 1 (weil backward alle Ausgabe-grad auf 1 setzt)
        for (i in betaVal.indices) {
            assertThat(beta.grad[i]).isCloseTo(1.0, offset(1e-9))
        }
    }
}

