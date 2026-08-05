package ch.zuegi.ml.llm.autograd

import java.util.Random

/**
 * Trainierbares minimales GPT-Modell auf Basis von [Tensor].
 *
 * End-to-End-Forward:
 *
 *     tokenIds
 *       -> Token-Embedding (Tabelle) + Positional-Embedding   [ctx, embeddingDim]
 *       -> N x TransformerBlockLayer                          [ctx, embeddingDim]
 *       -> finale LayerNorm (pro Zeile)                       [ctx, embeddingDim]
 *       -> Output-Projektion Wout                             [ctx, vocabSize]
 *
 * Alle Gewichte sind lernbare [Tensor]-Parameter (via [parameters] fuer [SGD]).
 * [loss] berechnet den mittleren Cross-Entropy ueber die Ziel-Tokens.
 *
 * @param vocabSize Groesse des Vokabulars.
 * @param contextLength Anzahl Positionen pro Sequenz.
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param numLayers Anzahl gestapelter Bloecke.
 * @param numHeads Anzahl Attention-Koepfe pro Block.
 * @param dK Dimension pro Kopf, Standard embeddingDim / numHeads.
 * @param hiddenDim versteckte Dimension der Feed-Forward-Netze, Standard 4 * embeddingDim.
 * @param causal wenn true, maskiert die Attention die Zukunft.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class GPTModelLayer(
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

    // Token-Embedding-Tabelle [vocabSize, embeddingDim], flach row-major
    val tokenEmbedding: Tensor =
        Tensor(DoubleArray(vocabSize * embeddingDim) { rnd.nextGaussian() * EMBED_SCALE })

    // Positional-Embedding-Tabelle [contextLength, embeddingDim], flach row-major
    val positionalEmbedding: Tensor =
        Tensor(DoubleArray(contextLength * embeddingDim) { rnd.nextGaussian() * EMBED_SCALE })

    private val blocks: List<TransformerBlockLayer> =
        (0 until numLayers).map { layerIndex ->
            TransformerBlockLayer(
                embeddingDim = embeddingDim,
                numHeads = numHeads,
                dK = dK,
                hiddenDim = hiddenDim,
                causal = causal,
                seed = seed?.let { it + (layerIndex + 1) * BLOCK_SEED_OFFSET },
            )
        }

    private val finalNorm = LayerNormLayer(embeddingDim)

    // Output-Projektion [embeddingDim, vocabSize], flach row-major
    val wOutput: Tensor =
        Tensor(DoubleArray(embeddingDim * vocabSize) { rnd.nextGaussian() * INIT_SCALE })

    /**
     * Forward-Pass.
     *
     * @param tokenIds Token-ID-Sequenz der Laenge [contextLength].
     * @return Logits als Matrix-Tensor [ctx, vocabSize], row-major.
     */
    fun forward(tokenIds: List<Int>): Tensor {
        require(tokenIds.size == contextLength) {
            "tokenIds.size ${tokenIds.size} passt nicht zu contextLength $contextLength"
        }

        val rows =
            (0 until contextLength).map { pos ->
                val tok = tokenEmbedding.embeddingLookup(tokenIds[pos], embeddingDim)
                val position = positionalEmbedding.embeddingLookup(pos, embeddingDim)
                tok + position
            }
        var x = Tensor.stackRows(rows, embeddingDim)

        for (block in blocks) {
            x = block.forward(x, contextLength)
        }

        val normed =
            Tensor.stackRows(
                (0 until contextLength).map { finalNorm.forward(x.row(it, embeddingDim)) },
                embeddingDim,
            )

        return normed.matMul(wOutput, p = contextLength, q = embeddingDim, r = vocabSize)
    }

    /**
     * Mittlerer Cross-Entropy-Loss ueber die Ziel-Tokens.
     *
     * @param tokenIds Eingabe-Sequenz der Laenge [contextLength].
     * @param targetIds Ziel-Sequenz gleicher Laenge (naechstes Token pro Position).
     * @return Skalar-Tensor mit dem mittleren Loss.
     */
    fun loss(
        tokenIds: List<Int>,
        targetIds: List<Int>,
    ): Tensor {
        require(targetIds.size == contextLength) {
            "targetIds.size ${targetIds.size} passt nicht zu contextLength $contextLength"
        }

        val logits = forward(tokenIds)
        var total = logits.row(0, vocabSize).softmaxCrossEntropy(targetIds[0])
        for (pos in 1 until contextLength) {
            total = total + logits.row(pos, vocabSize).softmaxCrossEntropy(targetIds[pos])
        }
        return total.scale(contextLength.toDouble())
    }

    /**
     * Erzeugt autoregressiv neue Tokens (reines Forward, keine Gradienten).
     *
     * Pro Schritt wird das letzte Kontextfenster durch das Modell geschickt und
     * aus den Logits der letzten Position das naechste Token gewaehlt:
     * - greedy = true: immer das wahrscheinlichste Token (deterministisch)
     * - greedy = false: Sampling gemaess Softmax, skaliert ueber [temperature];
     *   optional nur aus den [topK] wahrscheinlichsten Tokens
     *
     * @param startIds Start-Sequenz, mindestens [contextLength] Tokens.
     * @param maxNewTokens Anzahl zu erzeugender Tokens.
     * @param temperature Skalierung der Logits vor dem Sampling, muss > 0 sein.
     * @param topK wenn > 0, wird nur aus den topK wahrscheinlichsten Tokens gezogen.
     * @param greedy wenn true, deterministisches Maximum statt Sampling.
     * @param generatorSeed optionaler Seed fuer reproduzierbares Sampling.
     * @return Start-Sequenz inklusive der erzeugten Tokens.
     */
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
            val logits = lastPositionLogits(forward(sequence.takeLast(contextLength)))
            val next =
                if (greedy) {
                    argmax(logits)
                } else {
                    sampleFromLogits(logits, temperature, topK, generator)
                }
            sequence.add(next)
        }
        return sequence
    }

    /** Extrahiert die Logits der letzten Position als DoubleArray der Laenge vocabSize. */
    private fun lastPositionLogits(logits: Tensor): DoubleArray {
        val base = (contextLength - 1) * vocabSize
        return DoubleArray(vocabSize) { logits.data[base + it] }
    }

    private fun argmax(values: DoubleArray): Int {
        var maxIndex = 0
        for (i in 1 until values.size) {
            if (values[i] > values[maxIndex]) maxIndex = i
        }
        return maxIndex
    }

    /**
     * Zieht ein Token gemaess Softmax-Wahrscheinlichkeiten. Bei [topK] > 0 werden
     * nur die topK wahrscheinlichsten Logits beruecksichtigt, der Rest auf 0 gesetzt.
     */
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

    /** Behaelt nur die [k] groessten Logits, setzt den Rest auf -inf. */
    private fun keepTopK(
        logits: DoubleArray,
        k: Int,
    ): DoubleArray {
        val threshold = logits.sortedDescending()[k - 1]
        return DoubleArray(logits.size) { if (logits[it] >= threshold) logits[it] else Double.NEGATIVE_INFINITY }
    }

    private fun softmaxWithTemperature(
        logits: DoubleArray,
        temperature: Double,
    ): DoubleArray {
        val scaled = DoubleArray(logits.size) { logits[it] / temperature }
        val max = scaled.max()
        val exps = DoubleArray(scaled.size) { kotlin.math.exp(scaled[it] - max) }
        val sum = exps.sum()
        return DoubleArray(exps.size) { exps[it] / sum }
    }

    fun parameters(): List<Tensor> =
        listOf(tokenEmbedding, positionalEmbedding) +
            blocks.flatMap { it.parameters() } +
            finalNorm.parameters() +
            wOutput

    companion object {
        private const val INIT_SCALE = 0.02
        private const val EMBED_SCALE = 0.01
        private const val BLOCK_SEED_OFFSET = 100L
    }
}
