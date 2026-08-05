package ch.zuegi.ml.llm

import java.util.Random
import kotlin.math.exp

/**
 * Minimales GPT-Modell (nur Forward-Pass).
 *
 * Setzt die vorhandenen Bausteine zu einem End-to-End-Forward zusammen:
 *
 *     tokenIds
 *       -> InputEmbedding (Token + Positional)   [contextLength, embeddingDim]
 *       -> N x TransformerBlock                   [contextLength, embeddingDim]
 *       -> finale LayerNorm                       [contextLength, embeddingDim]
 *       -> Output-Projektion (Wout)               [contextLength, vocabSize]
 *
 * Der Output sind Logits pro Position ueber das gesamte Vokabular. Aus ihnen
 * wird spaeter via Softmax die Wahrscheinlichkeit des naechsten Tokens.
 *
 * @param vocabSize Groesse des Vokabulars.
 * @param contextLength Anzahl Positionen pro Sequenz.
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param numLayers Anzahl gestapelter Transformer-Bloecke.
 * @param numHeads Anzahl Attention-Koepfe pro Block.
 * @param dK Dimension pro Kopf, Standard embeddingDim / numHeads.
 * @param hiddenDim versteckte Dimension der Feed-Forward-Netze, Standard 4 * embeddingDim.
 * @param causal wenn true, maskiert die Attention die Zukunft (fuer Sprachmodelle ueblich).
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class GPTModel(
    private val vocabSize: Int,
    private val contextLength: Int,
    private val embeddingDim: Int,
    numLayers: Int,
    numHeads: Int,
    dK: Int = embeddingDim / numHeads,
    hiddenDim: Int = 4 * embeddingDim,
    causal: Boolean = true,
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
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    private val inputEmbedding =
        InputEmbedding(
            tokenEmbedding = TokenEmbedding(vocabSize, embeddingDim, seed = seed),
            positionalEmbedding =
                PositionalEmbedding(
                    contextLength,
                    embeddingDim,
                    seed = seed?.let { it + POSITIONAL_SEED_OFFSET },
                ),
        )

    private val blocks: List<TransformerBlock> =
        (0 until numLayers).map { layerIndex ->
            TransformerBlock(
                embeddingDim = embeddingDim,
                numHeads = numHeads,
                dK = dK,
                hiddenDim = hiddenDim,
                causal = causal,
                seed = seed?.let { it + (layerIndex + 1) * BLOCK_SEED_OFFSET },
            )
        }

    private val finalNorm = LayerNorm(embeddingDim)

    /**
     * Output-Projektion `[embeddingDim, vocabSize]` auf die Vokabular-Logits.
     */
    val wOutput: Array<DoubleArray> =
        Array(embeddingDim) { DoubleArray(vocabSize) { rnd.nextGaussian() * INIT_SCALE } }

    /**
     * Forward-Pass des Modells.
     *
     * @param tokenIds Token-ID-Sequenz der Laenge [contextLength].
     * @return Logits der Form `[contextLength, vocabSize]`.
     */
    fun forward(tokenIds: List<Int>): Array<DoubleArray> {
        require(tokenIds.size == contextLength) {
            "tokenIds.size ${tokenIds.size} passt nicht zu contextLength $contextLength"
        }

        var x = inputEmbedding.forward(tokenIds)
        for (block in blocks) {
            x = block.forward(x)
        }
        x = finalNorm.forward(x)

        return matMul(x, wOutput)
    }

    /**
     * Erzeugt autoregressiv neue Tokens.
     *
     * Pro Schritt wird das aktuelle Kontextfenster (die letzten [contextLength]
     * Tokens) durch das Modell geschickt, aus den Logits der letzten Position
     * das naechste Token bestimmt und angehaengt.
     *
     * Auswahlstrategie:
     * - greedy = true: immer das wahrscheinlichste Token (deterministisch)
     * - greedy = false: Sampling gemaess Softmax-Wahrscheinlichkeiten, skaliert
     *   ueber [temperature] (kleiner = konservativer, groesser = zufaelliger)
     *
     * @param startIds Start-Sequenz, mindestens [contextLength] Tokens.
     * @param maxNewTokens Anzahl zu erzeugender Tokens.
     * @param temperature Skalierung der Logits vor dem Sampling, muss > 0 sein.
     * @param greedy wenn true, wird deterministisch das Maximum gewaehlt.
     * @param generatorSeed optionaler Seed fuer reproduzierbares Sampling.
     * @return Start-Sequenz inklusive der erzeugten Tokens.
     */
    fun generate(
        startIds: List<Int>,
        maxNewTokens: Int,
        temperature: Double = 1.0,
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
            val window = sequence.takeLast(contextLength)
            val logits = forward(window)
            val lastLogits = logits[logits.size - 1]

            val nextToken =
                if (greedy) {
                    argmax(lastLogits)
                } else {
                    sampleFromLogits(lastLogits, temperature, generator)
                }

            sequence.add(nextToken)
        }

        return sequence
    }

    private fun argmax(values: DoubleArray): Int {
        var maxIndex = 0
        for (i in 1 until values.size) {
            if (values[i] > values[maxIndex]) {
                maxIndex = i
            }
        }
        return maxIndex
    }

    private fun sampleFromLogits(
        logits: DoubleArray,
        temperature: Double,
        generator: Random,
    ): Int {
        val probs = softmaxWithTemperature(logits, temperature)

        val threshold = generator.nextDouble()
        var cumulative = 0.0
        for (i in probs.indices) {
            cumulative += probs[i]
            if (threshold < cumulative) {
                return i
            }
        }
        return probs.size - 1
    }

    private fun softmaxWithTemperature(
        logits: DoubleArray,
        temperature: Double,
    ): DoubleArray {
        val scaled = DoubleArray(logits.size) { logits[it] / temperature }
        val max = scaled.max()
        val exps = DoubleArray(scaled.size) { exp(scaled[it] - max) }
        val sum = exps.sum()
        return DoubleArray(exps.size) { exps[it] / sum }
    }

    private fun matMul(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val rows = a.size
        val inner = b.size
        val cols = b[0].size

        return Array(rows) { i ->
            DoubleArray(cols) { j ->
                var sum = 0.0
                for (k in 0 until inner) {
                    sum += a[i][k] * b[k][j]
                }
                sum
            }
        }
    }

    companion object {
        private const val INIT_SCALE = 0.02
        private const val POSITIONAL_SEED_OFFSET = 500L
        private const val BLOCK_SEED_OFFSET = 100L
    }
}
