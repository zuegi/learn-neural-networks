package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class TransformerBlockLayerTest {
    @Test
    fun `output shape equals input shape`() {
        val block = TransformerBlockLayer(embeddingDim = 4, numHeads = 2, dK = 3, seed = 1)
        val input = Tensor(DoubleArray(3 * 4) { 0.05 * it })

        val out = block.forward(input, ctx = 3)

        assertThat(out.size).isEqualTo(3 * 4)
    }

    @Test
    fun `parameters include all submodules`() {
        val block = TransformerBlockLayer(embeddingDim = 4, numHeads = 2, dK = 3, seed = 1)

        val params = block.parameters()

        // 2 (ln1) + [2 Koepfe * 3 + Wo = 7] (attn) + 2 (ln2) + 4 (ff) = 15
        assertThat(params).hasSize(2 + 7 + 2 + 4)
    }

    @Test
    fun `training reduces loss towards a target matrix`() {
        val ctx = 2
        val dim = 4
        val block = TransformerBlockLayer(embeddingDim = dim, numHeads = 2, dK = 3, seed = 42)
        val sgd = SGD(block.parameters(), learningRate = 0.01)

        val input = Tensor(doubleArrayOf(0.5, -0.3, 0.8, -0.1, 0.2, 0.4, -0.6, 0.1))
        val target = doubleArrayOf(0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0, 0.3)
        val negTarget = Tensor(DoubleArray(ctx * dim) { -target[it] })
        val ones = Tensor(DoubleArray(ctx * dim) { 1.0 })

        fun currentLoss(): Double {
            val y = block.forward(input, ctx)
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
            val y = block.forward(input, ctx)
            val diff = y + negTarget
            val sq = diff * diff
            val loss = sq.matVecMul(ones, m = 1, n = ctx * dim)
            loss.backward()
            sgd.step()
        }

        assertThat(currentLoss()).isLessThan(initialLoss)
    }
}
