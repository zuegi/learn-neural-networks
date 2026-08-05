package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class MultiHeadAttentionLayerTest {
    @Test
    fun `output shape is ctx times embeddingDim`() {
        val mha = MultiHeadAttentionLayer(embeddingDim = 4, numHeads = 2, dK = 3, seed = 1)
        val input = Tensor(DoubleArray(2 * 4) { 0.1 * it })

        val out = mha.forward(input, ctx = 2)

        assertThat(out.size).isEqualTo(2 * 4)
    }

    @Test
    fun `parameters include all heads and output projection`() {
        val mha = MultiHeadAttentionLayer(embeddingDim = 4, numHeads = 2, dK = 3, seed = 1)

        val params = mha.parameters()

        // 3 Gewichte je Kopf * 2 Koepfe + Wo = 7
        assertThat(params).hasSize(3 * 2 + 1)
        assertThat(params.last()).isSameAs(mha.wOutput)
        assertThat(mha.wOutput.size).isEqualTo(2 * 3 * 4)
    }

    @Test
    fun `training reduces loss towards a target matrix`() {
        val ctx = 2
        val dim = 4
        val mha = MultiHeadAttentionLayer(embeddingDim = dim, numHeads = 2, dK = 3, seed = 42)
        val sgd = SGD(mha.parameters(), learningRate = 0.05)

        val input = Tensor(doubleArrayOf(0.5, -0.3, 0.8, -0.1, 0.2, 0.4, -0.6, 0.1))
        val target = doubleArrayOf(0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0, 0.3)
        val negTarget = Tensor(DoubleArray(ctx * dim) { -target[it] })
        val ones = Tensor(DoubleArray(ctx * dim) { 1.0 })

        fun currentLoss(): Double {
            val y = mha.forward(input, ctx)
            var total = 0.0
            for (i in target.indices) {
                val d = y.data[i] - target[i]
                total += d * d
            }
            return total
        }

        val initialLoss = currentLoss()

        repeat(300) {
            sgd.zeroGrad()
            val y = mha.forward(input, ctx)
            val diff = y + negTarget
            val sq = diff * diff
            val loss = sq.matVecMul(ones, m = 1, n = ctx * dim)
            loss.backward()
            sgd.step()
        }

        assertThat(currentLoss()).isLessThan(initialLoss)
    }
}
