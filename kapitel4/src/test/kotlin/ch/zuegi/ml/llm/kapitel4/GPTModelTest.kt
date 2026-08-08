package ch.zuegi.ml.llm.kapitel4

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class GPTModelTest {
    @Test
    fun `forward liefert logits mit shape contextLength mal vocabSize`() {
        val config =
            GPTConfig(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 2,
                numHeads = 2,
                seed = 42L,
            )
        val model = GPTModel(config)

        val tokenIds = listOf(1, 2, 3, 4)
        val logits = model.forward(tokenIds)

        assertEquals(config.contextLength * config.vocabSize, logits.size)
    }

    @Test
    fun `loss liefert skalar und ist finite`() {
        val config =
            GPTConfig(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 7L,
            )
        val model = GPTModel(config)

        val tokenIds = listOf(1, 2, 3, 4)
        val targetIds = listOf(2, 3, 4, 5)

        val loss = model.loss(tokenIds, targetIds)

        assertEquals(1, loss.size)
        assertTrue(loss.data[0].isFinite())
    }

    @Test
    fun `backward setzt gradients auf parametern`() {
        val config =
            GPTConfig(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 11L,
            )
        val model = GPTModel(config)

        val tokenIds = listOf(1, 2, 3, 4)
        val targetIds = listOf(2, 3, 4, 5)

        val loss = model.loss(tokenIds, targetIds)
        loss.backward()

        val params = model.parameters()
        assertFalse(params.isEmpty())
        assertTrue(params.any { p -> p.grad.any { it != 0.0 } })
    }

    @Test
    fun `generate liefert start plus maxNewTokens`() {
        val config =
            GPTConfig(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 21L,
            )
        val model = GPTModel(config)

        val startIds = listOf(1, 2, 3, 4)
        val generated = model.generate(startIds, maxNewTokens = 6, greedy = true)

        assertEquals(startIds.size + 6, generated.size)
    }

    @Test
    fun `generate ist reproduzierbar mit generatorSeed`() {
        val config =
            GPTConfig(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 99L,
            )
        val model = GPTModel(config)

        val startIds = listOf(1, 2, 3, 4)

        val g1 = model.generate(startIds, maxNewTokens = 8, greedy = false, generatorSeed = 123L)
        val g2 = model.generate(startIds, maxNewTokens = 8, greedy = false, generatorSeed = 123L)

        assertEquals(g1, g2)
    }

    @Test
    fun `forward wirft exception bei falscher contextLength`() {
        val config =
            GPTConfig(
                vocabSize = 20,
                contextLength = 4,
                embeddingDim = 8,
                numLayers = 1,
                numHeads = 2,
                seed = 42L,
            )
        val model = GPTModel(config)

        val ex =
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException::class.java) {
                model.forward(listOf(1, 2, 3)) // zu kurz
            }

        assertEquals(
            "tokenIds.size 3 passt nicht zu contextLength 4",
            ex.message,
        )
    }
}
