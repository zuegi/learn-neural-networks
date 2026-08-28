package ch.zuegi.ml.llm.shared.embedding

import java.util.Random

/**
 * Lernbare Token-Embedding-Tabelle.
 */
class TokenEmbedding(
    private val vocabSize: Int,
    private val embeddingDim: Int,
    seed: Long? = null,
) {
    init {
        require(vocabSize > 0) { "vocabSize muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val weights: Array<DoubleArray> =
        Array(vocabSize) { DoubleArray(embeddingDim) { rnd.nextGaussian() * INIT_SCALE } }

    fun lookup(tokenId: Int): DoubleArray {
        require(tokenId in 0 until vocabSize) { "tokenId $tokenId ausserhalb 0 until $vocabSize" }
        return weights[tokenId]
    }

    fun lookup(tokenIds: List<Int>): Array<DoubleArray> = Array(tokenIds.size) { lookup(tokenIds[it]) }

    companion object {
        private const val INIT_SCALE = 0.01
    }
}

