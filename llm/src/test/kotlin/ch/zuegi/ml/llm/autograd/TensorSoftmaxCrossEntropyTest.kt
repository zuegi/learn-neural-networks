package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.exp
import kotlin.math.ln

class TensorSoftmaxCrossEntropyTest {
    private fun lossOf(
        logits: DoubleArray,
        target: Int,
    ): Double {
        val max = logits.max()
        val exps = DoubleArray(logits.size) { exp(logits[it] - max) }
        val sum = exps.sum()
        return -ln(exps[target] / sum)
    }

    @Test
    fun `loss is positive and correct`() {
        val logits = Tensor(doubleArrayOf(2.0, 1.0, 0.1))

        val loss = logits.softmaxCrossEntropy(target = 0)

        assertThat(loss.data.size).isEqualTo(1)
        assertThat(loss.data[0]).isCloseTo(lossOf(doubleArrayOf(2.0, 1.0, 0.1), 0), offset(1e-12))
        assertThat(loss.data[0]).isGreaterThan(0.0)
    }

    @Test
    fun `confident correct prediction has near zero loss`() {
        val logits = Tensor(doubleArrayOf(100.0, 0.0, 0.0))

        val loss = logits.softmaxCrossEntropy(target = 0)

        assertThat(loss.data[0]).isCloseTo(0.0, offset(1e-6))
    }

    @Test
    fun `gradient equals softmax minus onehot`() {
        val logitsVal = doubleArrayOf(1.5, -0.5, 0.7)
        val target = 2

        val logits = Tensor(logitsVal)
        val loss = logits.softmaxCrossEntropy(target)
        loss.backward()

        // erwarteter Gradient: softmax - oneHot
        val max = logitsVal.max()
        val exps = DoubleArray(3) { exp(logitsVal[it] - max) }
        val sum = exps.sum()
        val probs = DoubleArray(3) { exps[it] / sum }

        for (i in 0 until 3) {
            val expected = probs[i] - if (i == target) 1.0 else 0.0
            assertThat(logits.grad[i]).isCloseTo(expected, offset(1e-12))
        }
    }

    @Test
    fun `gradient matches numeric gradient`() {
        val logitsVal = doubleArrayOf(0.8, -1.2, 0.3, 2.1)
        val target = 1

        val logits = Tensor(logitsVal)
        val loss = logits.softmaxCrossEntropy(target)
        loss.backward()

        val h = 1e-6
        for (i in logitsVal.indices) {
            val plus = logitsVal.copyOf().also { it[i] += h }
            val minus = logitsVal.copyOf().also { it[i] -= h }
            val numeric = (lossOf(plus, target) - lossOf(minus, target)) / (2 * h)
            assertThat(logits.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }

    @Test
    fun `chained with matVecMul matches numeric`() {
        // logits = W · x, dann Loss
        val xVal = doubleArrayOf(0.5, -0.3)
        val wVal = doubleArrayOf(1.0, 0.5, -0.2, 0.8, 0.3, -0.6) // 3x2
        val target = 2

        fun lossForX(x: DoubleArray): Double {
            val m = 3
            val n = 2
            val logits =
                DoubleArray(m) { row ->
                    var sum = 0.0
                    for (col in 0 until n) sum += wVal[row * n + col] * x[col]
                    sum
                }
            return lossOf(logits, target)
        }

        val x = Tensor(xVal)
        val w = Tensor(wVal)
        val logits = x.matVecMul(w, m = 3, n = 2)
        val loss = logits.softmaxCrossEntropy(target)
        loss.backward()

        val h = 1e-6
        for (i in xVal.indices) {
            val plus = xVal.copyOf().also { it[i] += h }
            val minus = xVal.copyOf().also { it[i] -= h }
            val numeric = (lossForX(plus) - lossForX(minus)) / (2 * h)
            assertThat(x.grad[i]).isCloseTo(numeric, offset(1e-6))
        }
    }
}
