package ch.zuegi.ml.llm.autograd

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class GPTModelLayerTest {
    private fun model() =
        GPTModelLayer(
            vocabSize = 7,
            contextLength = 3,
            embeddingDim = 4,
            numLayers = 2,
            numHeads = 2,
            dK = 2,
            seed = 42,
        )

    @Test
    fun `logits shape is ctx times vocabSize`() {
        val logits = model().forward(listOf(1, 2, 3))

        assertThat(logits.size).isEqualTo(3 * 7)
    }

    @Test
    fun `overfits a single sequence`() {
        val gpt = model()
        val sgd = SGD(gpt.parameters(), learningRate = 0.1)

        val input = listOf(1, 2, 3)
        val target = listOf(2, 3, 4)

        val initialLoss = gpt.loss(input, target).data[0]

        repeat(200) {
            sgd.zeroGrad()
            val loss = gpt.loss(input, target)
            loss.backward()
            sgd.step()
        }

        val finalLoss = gpt.loss(input, target).data[0]

        assertThat(finalLoss).isLessThan(initialLoss)
        assertThat(finalLoss).isLessThan(initialLoss * 0.5)
    }

    @Test
    fun `generate appends requested number of tokens`() {
        val gpt = model()

        val out = gpt.generate(listOf(1, 2, 3), maxNewTokens = 5)

        assertThat(out).hasSize(3 + 5)
        assertThat(out.take(3)).containsExactly(1, 2, 3)
        assertThat(out.subList(3, out.size).all { it in 0 until 7 }).isTrue()
    }
}
