package ch.zuegi.ml.llm.kapitel5.model

import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.ln

class GPTModelMultikTensorTest {
    private val config =
        GPTConfig(
            vocabSize = VOCAB_SIZE,
            contextLength = CONTEXT_LENGTH,
            embeddingDim = 8,
            numLayers = 2,
            numHeads = 2,
            dropoutProb = 0.0,
            seed = 7L,
        )
    private val tokenIds = listOf(1, 2, 3, 4)
    private val targetIds = listOf(2, 3, 4, 5)

    // tag::gpt-forward-shape[]
    @Test
    fun `forward liefert logits der Form contextLength mal vocabSize`() {
        val model = GPTModelMultikTensor(config)

        val logits = model.forward(tokenIds)

        assertEquals(CONTEXT_LENGTH * VOCAB_SIZE, logits.size)
    }
    // end::gpt-forward-shape[]

    @Test
    fun `Parameterzahl entspricht der Formel aus Kapitel 06`() {
        val model = GPTModelMultikTensor(config)
        val d = config.embeddingDim
        val perBlock = 2 * d + 4 * d * d + 2 * d + 2 * d * config.hiddenDim + config.hiddenDim + d
        val expected = 2 * VOCAB_SIZE * d + CONTEXT_LENGTH * d + config.numLayers * perBlock + 2 * d

        val actual = model.parameters().sumOf { it.size }

        assertEquals(expected, actual)
        assertEquals(EXPECTED_PARAMETER_COUNT, actual)
    }

    @Test
    fun `initialer loss liegt nahe ln vocabSize`() {
        val model = GPTModelMultikTensor(config)

        val loss = model.loss(tokenIds, targetIds).data[0]

        assertEquals(ln(VOCAB_SIZE.toDouble()), loss, INITIAL_LOSS_TOLERANCE)
    }

    @Test
    fun `backward erreicht jeden Parameter inklusive Embeddings und LayerNorm`() {
        val model = GPTModelMultikTensor(config)

        model.loss(tokenIds, targetIds).backward()

        model.parameters().forEachIndexed { index, parameter ->
            val hasGradient = (0 until parameter.size).any { parameter.grad[it] != 0.0 }
            assertTrue(hasGradient) { "Parameter $index (size ${parameter.size}) hat keinen Gradienten" }
        }
    }

    @Test
    fun `kausale Maske verhindert Einfluss zukuenftiger Token auf fruehere Logits`() {
        val model = GPTModelMultikTensor(config)

        val original = model.forward(listOf(1, 2, 3, 4))
        val changedFuture = model.forward(listOf(1, 2, 3, 9))

        for (i in 0 until (CONTEXT_LENGTH - 1) * VOCAB_SIZE) {
            assertEquals(original.data[i], changedFuture.data[i], EXACT_TOLERANCE)
        }
    }

    @Test
    fun `generate haengt maxNewTokens an und ist mit generatorSeed reproduzierbar`() {
        val model = GPTModelMultikTensor(config)
        val generationConfig = GenerationConfig(maxNewTokens = 3, topK = 5, generatorSeed = 5L)

        val first = model.generate(tokenIds, generationConfig)
        val second = model.generate(tokenIds, generationConfig)

        assertEquals(tokenIds.size + 3, first.size)
        assertEquals(tokenIds, first.take(tokenIds.size))
        assertEquals(first, second)
    }

    companion object {
        private const val VOCAB_SIZE = 12
        private const val CONTEXT_LENGTH = 4
        private const val INITIAL_LOSS_TOLERANCE = 0.1
        private const val EXACT_TOLERANCE = 1e-12
        private const val EXPECTED_PARAMETER_COUNT = 1920
    }
}
