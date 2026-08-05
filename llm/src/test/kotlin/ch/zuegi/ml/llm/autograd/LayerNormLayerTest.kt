package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class LayerNormLayerTest {
    @Test
    fun `parameters expose gamma and beta`() {
        val layer = LayerNormLayer(embeddingDim = 4)

        val params = layer.parameters()

        assertThat(params).containsExactly(layer.gamma, layer.beta)
        assertThat(layer.gamma.data).containsOnly(1.0)
        assertThat(layer.beta.data).containsOnly(0.0)
    }

    @Test
    fun `forward normalizes to mean zero variance one initially`() {
        val layer = LayerNormLayer(embeddingDim = 4)

        val y = layer.forward(Tensor(doubleArrayOf(1.0, 2.0, 3.0, 4.0)))

        val mean = y.data.average()
        val variance = y.data.sumOf { (it - mean) * (it - mean) } / y.size
        assertThat(mean).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-9))
        assertThat(variance).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-4))
    }

    @Test
    fun `training adjusts gamma and beta to reduce loss`() {
        val n = 4
        val layer = LayerNormLayer(embeddingDim = n)
        val sgd = SGD(layer.parameters(), learningRate = 0.1)

        // Ziel: konstante Zielwerte, die nur ueber gamma/beta erreichbar sind
        val input = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val target = doubleArrayOf(0.5, -0.5, 0.5, -0.5)

        fun loss(): Double {
            val y = layer.forward(Tensor(input))
            var total = 0.0
            for (i in 0 until n) {
                val diff = y.data[i] - target[i]
                total += diff * diff
            }
            return total
        }

        val initialLoss = loss()

        repeat(200) {
            sgd.zeroGrad()
            val y = layer.forward(Tensor(input))

            // MSE-Loss als Tensor-Graph: Σ (y - target)^2
            // via (y + (-target)) elementweise, dann Quadrat durch times mit sich selbst
            val negTarget = Tensor(DoubleArray(n) { -target[it] })
            val diff = y + negTarget
            val sq = diff * diff

            // Summe der Komponenten: baue Skalar durch matVecMul mit Einsen-Zeile [1, n]
            val ones = Tensor(DoubleArray(n) { 1.0 })
            val lossTensor = sq.matVecMul(ones, m = 1, n = n)

            lossTensor.backward()
            sgd.step()
        }

        val finalLoss = loss()

        assertThat(finalLoss).isLessThan(initialLoss)
        assertThat(finalLoss).isLessThan(initialLoss * 0.5)
    }
}

