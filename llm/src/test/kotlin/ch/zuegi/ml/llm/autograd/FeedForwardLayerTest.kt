package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class FeedForwardLayerTest {
    @Test
    fun `output has same length as input`() {
        val ff = FeedForwardLayer(embeddingDim = 4, seed = 42)

        val y = ff.forward(Tensor(doubleArrayOf(0.1, -0.2, 0.3, 0.4)))

        assertThat(y.size).isEqualTo(4)
    }

    @Test
    fun `parameters expose all weights and biases`() {
        val ff = FeedForwardLayer(embeddingDim = 4, hiddenDim = 8, seed = 42)

        val params = ff.parameters()

        assertThat(params).containsExactly(ff.w1, ff.b1, ff.w2, ff.b2)
        assertThat(ff.w1.size).isEqualTo(8 * 4)
        assertThat(ff.w2.size).isEqualTo(4 * 8)
        assertThat(ff.b1.size).isEqualTo(8)
        assertThat(ff.b2.size).isEqualTo(4)
    }

    @Test
    fun `training reduces loss towards a target vector`() {
        val n = 4
        val ff = FeedForwardLayer(embeddingDim = n, hiddenDim = 16, seed = 42)
        val sgd = SGD(ff.parameters(), learningRate = 0.05)

        val input = doubleArrayOf(0.5, -0.3, 0.8, -0.1)
        val target = doubleArrayOf(1.0, 0.0, -1.0, 0.5)
        val ones = Tensor(DoubleArray(n) { 1.0 })
        val negTarget = Tensor(DoubleArray(n) { -target[it] })

        fun currentLoss(): Double {
            val y = ff.forward(Tensor(input))
            var total = 0.0
            for (i in 0 until n) {
                val d = y.data[i] - target[i]
                total += d * d
            }
            return total
        }

        val initialLoss = currentLoss()

        repeat(300) {
            sgd.zeroGrad()
            val y = ff.forward(Tensor(input))
            val diff = y + negTarget
            val sq = diff * diff
            val loss = sq.matVecMul(ones, m = 1, n = n)
            loss.backward()
            sgd.step()
        }

        val finalLoss = currentLoss()

        assertThat(finalLoss).isLessThan(initialLoss)
        assertThat(finalLoss).isLessThan(initialLoss * 0.2)
    }
}
