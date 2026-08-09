package ch.zuegi.ml.llm.kapitel4

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class GPTModelMultikTest {
    @Test
    fun `forward liefert logits mit shape contextLength mal vocabSize`() {
        val model =
            GPTModelMultik(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 2,
                numHeads = 2,
                seed = 42L,
            )

        val logits = model.forward(listOf(1, 2, 3, 4), training = false)

        assertEquals(4, logits.shape[0])
        assertEquals(20, logits.shape[1])
    }

    @Test
    fun `loss liefert finite zahl`() {
        val model =
            GPTModelMultik(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 7L,
            )

        val loss = model.loss(tokenIds = listOf(1, 2, 3, 4), targetIds = listOf(2, 3, 4, 5))

        assertTrue(loss.isFinite())
    }

    @Test
    fun `generate liefert start plus maxNewTokens`() {
        val model =
            GPTModelMultik(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 21L,
            )

        val start = listOf(1, 2, 3, 4)
        val generated = model.generate(start, maxNewTokens = 6, greedy = true)

        assertEquals(start.size + 6, generated.size)
    }

    @Test
    fun `generate ist reproduzierbar mit generatorSeed`() {
        val model =
            GPTModelMultik(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 99L,
            )

        val start = listOf(1, 2, 3, 4)

        val g1 = model.generate(start, maxNewTokens = 8, greedy = false, generatorSeed = 123L)
        val g2 = model.generate(start, maxNewTokens = 8, greedy = false, generatorSeed = 123L)

        assertEquals(g1, g2)
    }
}
