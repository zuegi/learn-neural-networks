package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class GPTModelTest {
    private val vocabSize = 20
    private val contextLength = 4
    private val embeddingDim = 8

    private fun model(seed: Long? = 42) =
        GPTModel(
            vocabSize = vocabSize,
            contextLength = contextLength,
            embeddingDim = embeddingDim,
            numLayers = 2,
            numHeads = 2,
            seed = seed,
        )

    @Test
    fun `forward returns logits of shape contextLength x vocabSize`() {
        val logits = model().forward(listOf(1, 2, 3, 4))

        assertThat(logits.size).isEqualTo(contextLength)
        assertThat(logits.all { it.size == vocabSize }).isTrue()
    }

    @Test
    fun `same seed produces same logits`() {
        val a = model(seed = 42).forward(listOf(1, 2, 3, 4))
        val b = model(seed = 42).forward(listOf(1, 2, 3, 4))

        assertThat(a.contentDeepEquals(b)).isTrue()
    }

    @Test
    fun `forward rejects wrong sequence length`() {
        assertThatThrownBy { model().forward(listOf(1, 2, 3)) }
            .isInstanceOf(IllegalArgumentException::class.java)
    }
}

