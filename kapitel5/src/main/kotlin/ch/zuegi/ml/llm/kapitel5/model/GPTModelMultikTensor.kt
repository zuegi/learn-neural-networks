package ch.zuegi.ml.llm.kapitel5.model
import ch.zuegi.ml.llm.kapitel5.library.autograd.LayerNormMultikTensor
import ch.zuegi.ml.llm.kapitel5.library.autograd.TensorMultik
import ch.zuegi.ml.llm.kapitel5.library.autograd.TransformerBlockMultikTensor
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import java.util.Random
import kotlin.math.exp

class GPTModelMultikTensor(
    private val config: GPTConfig,
) {
    private val vocabSize = config.vocabSize
    private val contextLength = config.contextLength
    private val embeddingDim = config.embeddingDim

    private val rnd = if (config.seed != null) Random(config.seed) else Random()

    val tokenEmbedding: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(vocabSize * embeddingDim) { rnd.nextGaussian() * EMBED_SCALE }),
        )
    val positionalEmbedding: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(contextLength * embeddingDim) { rnd.nextGaussian() * EMBED_SCALE }),
        )

    private val blocks: List<TransformerBlockMultikTensor> =
        (0 until config.numLayers).map { layerIndex ->
            TransformerBlockMultikTensor(
                embeddingDim = embeddingDim,
                numHeads = config.numHeads,
                dK = config.dK,
                hiddenDim = config.hiddenDim,
                causal = config.causal,
                dropoutProb = config.dropoutProb,
                useQkvBias = config.useQkvBias,
                useOutputBias = config.useOutputBias,
                seed = config.seed?.let { it + (layerIndex + 1) * BLOCK_SEED_OFFSET },
            )
        }

    private val finalNorm = LayerNormMultikTensor(embeddingDim)
    val wOutput: TensorMultik =
        TensorMultik(
            mk.ndarray(DoubleArray(embeddingDim * vocabSize) { rnd.nextGaussian() * INIT_SCALE }),
        )

    fun forward(
        tokenIds: List<Int>,
        training: Boolean = false,
    ): TensorMultik {
        require(tokenIds.size == contextLength) {
            "tokenIds.size ${tokenIds.size} passt nicht zu contextLength $contextLength"
        }

        val rows =
            (0 until contextLength).map { pos ->
                val tok = tokenEmbedding.row(tokenIds[pos], embeddingDim)
                val position = positionalEmbedding.row(pos, embeddingDim)
                tok + position
            }
        var x = TensorMultik.stackRows(rows, embeddingDim)

        for (block in blocks) {
            x = block.forward(x, contextLength, training)
        }

        val normed =
            TensorMultik.stackRows(
                (0 until contextLength).map { finalNorm.forward(x.row(it, embeddingDim)) },
                embeddingDim,
            )

        return normed.matMul(wOutput, p = contextLength, q = embeddingDim, r = vocabSize)
    }

    fun loss(
        tokenIds: List<Int>,
        targetIds: List<Int>,
    ): TensorMultik {
        require(targetIds.size == contextLength) {
            "targetIds.size ${targetIds.size} passt nicht zu contextLength $contextLength"
        }

        val logits = forward(tokenIds, training = false)
        var total = logits.row(0, vocabSize).softmaxCrossEntropy(targetIds[0])
        for (pos in 1 until contextLength) {
            total = total + logits.row(pos, vocabSize).softmaxCrossEntropy(targetIds[pos])
        }
        return total.scale(contextLength.toDouble())
    }

    fun generate(
        startIds: List<Int>,
        generatedConfig: GenerationConfig,
    ): List<Int> {
        require(startIds.size >= contextLength) {
            "startIds.size ${startIds.size} muss >= contextLength $contextLength sein"
        }
        require(generatedConfig.maxNewTokens >= 0) { "maxNewTokens muss >= 0 sein" }
        require(generatedConfig.temperature > 0.0) { "temperature muss > 0 sein" }

        val generator = if (generatedConfig.generatorSeed != null) Random(generatedConfig.generatorSeed) else Random()
        val sequence = startIds.toMutableList()

        repeat(generatedConfig.maxNewTokens) {
            val logits = forward(sequence.takeLast(contextLength), training = false)
            val last = logits.row(contextLength - 1, vocabSize)
            val lastArray = DoubleArray(vocabSize) { i -> last.data[i] }

            val next =
                if (generatedConfig.greedy) {
                    argmax(lastArray)
                } else {
                    sampleFromLogits(
                        lastArray,
                        generatedConfig.temperature,
                        generatedConfig.topK,
                        generator,
                    )
                }
            sequence.add(next)
        }

        return sequence
    }

    private fun argmax(values: DoubleArray): Int {
        var maxIndex = 0
        for (i in 1 until values.size) {
            if (values[i] > values[maxIndex]) maxIndex = i
        }
        return maxIndex
    }

    private fun sampleFromLogits(
        logits: DoubleArray,
        temperature: Double,
        topK: Int,
        generator: Random,
    ): Int {
        val filtered = if (topK in 1 until logits.size) keepTopK(logits, topK) else logits
        val probs = softmaxWithTemperature(filtered, temperature)

        val threshold = generator.nextDouble()
        var cumulative = 0.0
        for (i in probs.indices) {
            cumulative += probs[i]
            if (threshold < cumulative) return i
        }
        return probs.size - 1
    }

    private fun keepTopK(
        logits: DoubleArray,
        k: Int,
    ): DoubleArray {
        val threshold = logits.sortedDescending()[k - 1]
        return DoubleArray(logits.size) { i ->
            if (logits[i] >= threshold) logits[i] else Double.NEGATIVE_INFINITY
        }
    }

    private fun softmaxWithTemperature(
        logits: DoubleArray,
        temperature: Double,
    ): DoubleArray {
        val scaled = DoubleArray(logits.size) { i -> logits[i] / temperature }
        val max = scaled.maxOrNull() ?: 0.0
        val exps = DoubleArray(scaled.size) { i -> exp(scaled[i] - max) }
        val sum = exps.sum()
        return DoubleArray(exps.size) { i -> exps[i] / sum }
    }

    fun parameters(): List<TensorMultik> =
        listOf(tokenEmbedding, positionalEmbedding) +
            blocks.flatMap { it.parameters() } +
            finalNorm.parameters() +
            listOf(wOutput)

    companion object {
        private const val INIT_SCALE = 0.02
        private const val EMBED_SCALE = 0.01
        private const val BLOCK_SEED_OFFSET = 100L
    }
}
