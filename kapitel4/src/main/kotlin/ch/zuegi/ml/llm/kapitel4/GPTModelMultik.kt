package ch.zuegi.ml.llm.kapitel4

import ch.zuegi.ml.llm.kapitel4.library.autograd.LayerNormMultik
import ch.zuegi.ml.llm.kapitel4.library.autograd.TransformerBlockMultik
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import java.util.Random
import kotlin.math.exp
import kotlin.math.ln

class GPTModelMultik(
    private val vocabSize: Int,
    private val contextLength: Int,
    private val embeddingDim: Int,
    numLayers: Int,
    numHeads: Int,
    dK: Int = embeddingDim / numHeads,
    hiddenDim: Int = 4 * embeddingDim,
    causal: Boolean = true,
    private val dropoutProb: Double = 0.0,
    seed: Long? = null,
) {
    init {
        require(vocabSize > 0) { "vocabSize muss > 0 sein" }
        require(contextLength > 0) { "contextLength muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(numLayers > 0) { "numLayers muss > 0 sein" }
        require(numHeads > 0) { "numHeads muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
        require(dropoutProb in 0.0..1.0) { "dropoutProb muss in [0.0, 1.0] liegen" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val tokenEmbedding: NDArray<Double, D2> = randomMatrix(vocabSize, embeddingDim, EMBED_SCALE)
    val positionalEmbedding: NDArray<Double, D2> = randomMatrix(contextLength, embeddingDim, EMBED_SCALE)

    private val blocks: List<TransformerBlockMultik> =
        (0 until numLayers).map { layerIndex ->
            TransformerBlockMultik(
                embeddingDim = embeddingDim,
                numHeads = numHeads,
                dK = dK,
                hiddenDim = hiddenDim,
                causal = causal,
                dropoutProb = dropoutProb,
                seed = seed?.let { it + (layerIndex + 1) * BLOCK_SEED_OFFSET },
            )
        }

    private val finalNorm = LayerNormMultik(embeddingDim)
    val wOutput: NDArray<Double, D2> = randomMatrix(embeddingDim, vocabSize, INIT_SCALE)

    fun forward(
        tokenIds: List<Int>,
        training: Boolean = false,
    ): NDArray<Double, D2> {
        require(tokenIds.size == contextLength) {
            "tokenIds.size ${tokenIds.size} passt nicht zu contextLength $contextLength"
        }

        var x =
            mk.ndarray(
                List(contextLength) { pos ->
                    List(embeddingDim) { dim ->
                        tokenEmbedding[tokenIds[pos]][dim] + positionalEmbedding[pos][dim]
                    }
                },
            )

        for (block in blocks) {
            x = block.forward(x, training)
        }

        val normed =
            mk.ndarray(
                List(contextLength) { pos ->
                    finalNorm.forward(row(x, pos)).toList()
                },
            )

        return matMul(normed, wOutput)
    }

    fun loss(
        tokenIds: List<Int>,
        targetIds: List<Int>,
    ): Double {
        require(targetIds.size == contextLength) {
            "targetIds.size ${targetIds.size} passt nicht zu contextLength $contextLength"
        }

        val logits = forward(tokenIds, training = false)

        var total = 0.0
        for (pos in 0 until contextLength) {
            total += softmaxCrossEntropy(row(logits, pos), targetIds[pos])
        }
        return total / contextLength
    }

    fun generate(
        startIds: List<Int>,
        maxNewTokens: Int,
        temperature: Double = 1.0,
        topK: Int = 0,
        greedy: Boolean = false,
        generatorSeed: Long? = null,
    ): List<Int> {
        require(startIds.size >= contextLength) {
            "startIds.size ${startIds.size} muss >= contextLength $contextLength sein"
        }
        require(maxNewTokens >= 0) { "maxNewTokens muss >= 0 sein" }
        require(temperature > 0.0) { "temperature muss > 0 sein" }

        val generator = if (generatorSeed != null) Random(generatorSeed) else Random()
        val sequence = startIds.toMutableList()

        repeat(maxNewTokens) {
            val logits = forward(sequence.takeLast(contextLength), training = false)
            val last = row(logits, contextLength - 1).toDoubleArray()

            val next =
                if (greedy) {
                    argmax(last)
                } else {
                    sampleFromLogits(last, temperature, topK, generator)
                }

            sequence.add(next)
        }

        return sequence
    }

    private fun softmaxCrossEntropy(
        logits: NDArray<Double, D1>,
        target: Int,
    ): Double {
        require(target in 0 until logits.size) { "target $target ausserhalb 0 until ${logits.size}" }
        val max = (0 until logits.size).maxOf { logits[it] }
        val exps = DoubleArray(logits.size) { i -> exp(logits[i] - max) }
        val sum = exps.sum()
        val probs = DoubleArray(logits.size) { i -> exps[i] / sum }
        return -ln(probs[target])
    }

    private fun argmax(values: DoubleArray): Int {
        var maxIndex = 0
        for (i in 1 until values.size) if (values[i] > values[maxIndex]) maxIndex = i
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

    private fun randomMatrix(
        rows: Int,
        cols: Int,
        scale: Double,
    ): NDArray<Double, D2> =
        mk.ndarray(
            List(rows) {
                List(cols) { rnd.nextGaussian() * scale }
            },
        )

    private fun matMul(
        a: NDArray<Double, D2>,
        b: NDArray<Double, D2>,
    ): NDArray<Double, D2> {
        require(a.shape[1] == b.shape[0]) {
            "Matrix-Dimensionen passen nicht: [${a.shape[0]}, ${a.shape[1]}] x [${b.shape[0]}, ${b.shape[1]}]"
        }

        val rows = a.shape[0]
        val inner = a.shape[1]
        val cols = b.shape[1]

        return mk.ndarray(
            List(rows) { i ->
                List(cols) { j ->
                    var sum = 0.0
                    for (k in 0 until inner) sum += a[i][k] * b[k][j]
                    sum
                }
            },
        )
    }

    private fun row(
        matrix: NDArray<Double, D2>,
        rowIndex: Int,
    ): NDArray<Double, D1> = mk.ndarray(DoubleArray(matrix.shape[1]) { c -> matrix[rowIndex][c] })

    private fun NDArray<Double, D1>.toDoubleArray(): DoubleArray = DoubleArray(size) { this[it] }

    companion object {
        private const val INIT_SCALE = 0.02
        private const val EMBED_SCALE = 0.01
        private const val BLOCK_SEED_OFFSET = 100L
    }
}
