package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LayerNormMultikTensorTest {
    private val inputValues = doubleArrayOf(0.5, -1.0, 2.0, 0.25)
    private val weights = doubleArrayOf(0.3, -0.7, 1.1, 0.2)

    @Test
    fun `forward normalisiert auf Mittelwert null und Varianz eins`() {
        val norm = LayerNormMultikTensor(inputValues.size)

        val out = norm.forward(TensorMultik(mk.ndarray(inputValues.copyOf())))

        val mean = (0 until out.size).sumOf { out.data[it] } / out.size
        val variance = (0 until out.size).sumOf { (out.data[it] - mean) * (out.data[it] - mean) } / out.size
        assertEquals(0.0, mean, STAT_TOLERANCE)
        assertEquals(1.0, variance, VARIANCE_TOLERANCE)
    }

    @Test
    fun `backward stimmt mit Finite-Differenzen fuer Eingabe gamma und beta ueberein`() {
        val norm = LayerNormMultikTensor(inputValues.size)
        val x = TensorMultik(mk.ndarray(inputValues.copyOf()))

        weightedSum(norm, x).backward()

        for (tensor in listOf(x, norm.gamma, norm.beta)) {
            for (i in 0 until tensor.size) {
                val numeric = numericGradient(norm, x, tensor, i)
                assertEquals(numeric, tensor.grad[i], GRADIENT_TOLERANCE)
            }
        }
    }

    private fun weightedSum(
        norm: LayerNormMultikTensor,
        x: TensorMultik,
    ): TensorMultik = norm.forward(x).dot(TensorMultik(mk.ndarray(weights.copyOf())))

    private fun numericGradient(
        norm: LayerNormMultikTensor,
        x: TensorMultik,
        tensor: TensorMultik,
        index: Int,
    ): Double {
        val original = tensor.data[index]
        tensor.data[index] = original + EPSILON
        val plus = weightedSum(norm, x).data[0]
        tensor.data[index] = original - EPSILON
        val minus = weightedSum(norm, x).data[0]
        tensor.data[index] = original
        return (plus - minus) / (2 * EPSILON)
    }

    companion object {
        private const val EPSILON = 1e-6
        private const val STAT_TOLERANCE = 1e-12
        private const val VARIANCE_TOLERANCE = 1e-4
        private const val GRADIENT_TOLERANCE = 1e-6
    }
}
