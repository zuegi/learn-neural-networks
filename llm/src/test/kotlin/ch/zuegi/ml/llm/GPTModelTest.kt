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

    @Test
    fun `generate appends maxNewTokens tokens`() {
        val start = listOf(1, 2, 3, 4)

        val result = model().generate(start, maxNewTokens = 5, greedy = true)

        assertThat(result.size).isEqualTo(start.size + 5)
        assertThat(result.subList(0, start.size)).isEqualTo(start)
    }

    @Test
    fun `generated tokens are within vocabulary range`() {
        val result = model().generate(listOf(1, 2, 3, 4), maxNewTokens = 10, greedy = true)

        assertThat(result.all { it in 0 until vocabSize }).isTrue()
    }

    @Test
    fun `greedy generation is deterministic`() {
        val a = model(seed = 42).generate(listOf(1, 2, 3, 4), maxNewTokens = 8, greedy = true)
        val b = model(seed = 42).generate(listOf(1, 2, 3, 4), maxNewTokens = 8, greedy = true)

        assertThat(a).isEqualTo(b)
    }

    @Test
    fun `sampling with same generator seed is reproducible`() {
        val a = model(seed = 42).generate(listOf(1, 2, 3, 4), maxNewTokens = 8, generatorSeed = 7)
        val b = model(seed = 42).generate(listOf(1, 2, 3, 4), maxNewTokens = 8, generatorSeed = 7)

        assertThat(a).isEqualTo(b)
    }

    @Test
    fun `generate rejects too short start sequence`() {
        assertThatThrownBy { model().generate(listOf(1, 2), maxNewTokens = 3) }
            .isInstanceOf(IllegalArgumentException::class.java)
    }
}
