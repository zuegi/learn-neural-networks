package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import java.util.Random

class SGDTest {
    @Test
    fun `step moves parameter against gradient`() {
        val param = Tensor(doubleArrayOf(1.0, 2.0))
        // grad manuell setzen
        param.grad[0] = 0.5
        param.grad[1] = -1.0

        val sgd = SGD(listOf(param), learningRate = 0.1)
        sgd.step()

        // data -= lr * grad
        assertThat(param.data[0]).isCloseTo(1.0 - 0.1 * 0.5, org.assertj.core.data.Offset.offset(1e-12))
        assertThat(param.data[1]).isCloseTo(2.0 - 0.1 * -1.0, org.assertj.core.data.Offset.offset(1e-12))
    }

    @Test
    fun `zeroGrad resets all parameter gradients`() {
        val a = Tensor(doubleArrayOf(1.0, 2.0))
        val b = Tensor(doubleArrayOf(3.0))
        a.grad[0] = 5.0
        b.grad[0] = 7.0

        SGD(listOf(a, b), learningRate = 0.1).zeroGrad()

        assertThat(a.grad).containsExactly(0.0, 0.0)
        assertThat(b.grad).containsExactly(0.0)
    }

    @Test
    fun `constructor rejects non positive learning rate`() {
        assertThatThrownBy { SGD(listOf(Tensor(doubleArrayOf(1.0))), learningRate = 0.0) }
            .isInstanceOf(IllegalArgumentException::class.java)
    }

    @Test
    fun `training reduces loss on a simple classification task`() {
        val rnd = Random(42)
        val m = 3 // Klassen
        val n = 4 // Eingabegroesse

        // lernbare Gewichtsmatrix [m, n], flach
        val w = Tensor(DoubleArray(m * n) { rnd.nextGaussian() * 0.1 })
        val sgd = SGD(listOf(w), learningRate = 0.5)

        // fixe Trainingsdaten: (Eingabe, Zielklasse)
        val samples =
            listOf(
                doubleArrayOf(1.0, 0.0, 0.0, 0.0) to 0,
                doubleArrayOf(0.0, 1.0, 0.0, 0.0) to 1,
                doubleArrayOf(0.0, 0.0, 1.0, 0.0) to 2,
            )

        fun epochLoss(): Double {
            var total = 0.0
            for ((xVal, target) in samples) {
                val x = Tensor(xVal)
                val logits = x.matVecMul(w, m = m, n = n)
                total += logits.softmaxCrossEntropy(target).data[0]
            }
            return total / samples.size
        }

        val initialLoss = epochLoss()

        repeat(100) {
            for ((xVal, target) in samples) {
                sgd.zeroGrad()
                val x = Tensor(xVal)
                val logits = x.matVecMul(w, m = m, n = n)
                val loss = logits.softmaxCrossEntropy(target)
                loss.backward()
                sgd.step()
            }
        }

        val finalLoss = epochLoss()

        assertThat(finalLoss).isLessThan(initialLoss)
        assertThat(finalLoss).isLessThan(0.5) // klar gelernt
    }
}

