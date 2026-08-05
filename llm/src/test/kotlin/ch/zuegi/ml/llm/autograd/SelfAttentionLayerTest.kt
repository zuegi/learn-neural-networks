package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.within
import org.junit.jupiter.api.Test

class SelfAttentionLayerTest {
    @Test
    fun `output shape is ctx times dK`() {
        val attn = SelfAttentionLayer(embeddingDim = 4, dK = 3, seed = 1)
        val input = Tensor(DoubleArray(2 * 4) { 0.1 * it })

        val out = attn.forward(input, ctx = 2)

        assertThat(out.size).isEqualTo(2 * 3)
    }

    @Test
    fun `causal last position sees full sequence`() {
        val causal = SelfAttentionLayer(embeddingDim = 4, dK = 3, causal = true, seed = 7)
        val full = SelfAttentionLayer(embeddingDim = 4, dK = 3, causal = false, seed = 7)
        val input = Tensor(DoubleArray(3 * 4) { 0.05 * it })

        val cOut = causal.forward(input, ctx = 3)
        val fOut = full.forward(input, ctx = 3)

        // letzte Position sieht in beiden Faellen die ganze Sequenz
        val dK = 3
        for (k in 0 until dK) {
            assertThat(cOut.data[2 * dK + k]).isCloseTo(fOut.data[2 * dK + k], within(1e-9))
        }
    }

    @Test
    fun `training reduces loss towards a target matrix`() {
        val ctx = 2
        val dK = 3
        val attn = SelfAttentionLayer(embeddingDim = 4, dK = dK, seed = 42)
        val sgd = SGD(attn.parameters(), learningRate = 0.05)

        val input = Tensor(doubleArrayOf(0.5, -0.3, 0.8, -0.1, 0.2, 0.4, -0.6, 0.1))
        val target = doubleArrayOf(0.1, -0.2, 0.3, 0.4, -0.1, 0.2)
        val negTarget = Tensor(DoubleArray(ctx * dK) { -target[it] })
        val ones = Tensor(DoubleArray(ctx * dK) { 1.0 })

        fun currentLoss(): Double {
            val y = attn.forward(input, ctx)
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
            val y = attn.forward(input, ctx)
            val diff = y + negTarget
            val sq = diff * diff
            val loss = sq.matVecMul(ones, m = 1, n = ctx * dK)
            loss.backward()
            sgd.step()
        }

        assertThat(currentLoss()).isLessThan(initialLoss)
    }
}
