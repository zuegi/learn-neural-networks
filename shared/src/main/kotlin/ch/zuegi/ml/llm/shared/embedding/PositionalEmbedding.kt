package ch.zuegi.ml.llm.shared.embedding

import java.util.Random

/**
 * Lernbare Positional-Embedding-Tabelle.
 */
class PositionalEmbedding(
    private val contextLength: Int,
    private val embeddingDim: Int,
    seed: Long? = null,
) {
    init {
        require(contextLength > 0) { "contextLength muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val weights: Array<DoubleArray> =
        Array(contextLength) { DoubleArray(embeddingDim) { rnd.nextGaussian() * INIT_SCALE } }

    fun lookup(position: Int): DoubleArray {
        require(position in 0 until contextLength) { "position $position ausserhalb 0 until $contextLength" }
        return weights[position]
    }

    fun lookupAll(): Array<DoubleArray> = weights

    companion object {
        private const val INIT_SCALE = 0.01
    }
}

