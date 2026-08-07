package ch.zuegi.ml.llm.kapitel3.scratch.embedding

import java.util.Random

/**
 * Lernbare Token-Embedding-Tabelle.
 *
 * Ordnet jeder Token-ID einen dichten Vektor der Länge [embeddingDim] zu.
 * Die Gewichte bilden eine Matrix der Form `[vocabSize, embeddingDim]`,
 * bei der jede Zeile das Embedding einer Token-ID ist.
 *
 * Das Nachschlagen ist ein reiner Zeilenzugriff, keine Matrixmultiplikation.
 *
 * @param vocabSize Anzahl Tokens im Vokabular.
 * @param embeddingDim Länge eines Embedding-Vektors.
 * @param seed optionaler Seed für reproduzierbare Initialisierung.
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

    /**
     * Embedding-Matrix `[vocabSize, embeddingDim]` mit kleinen Zufallswerten.
     */
    val weights: Array<DoubleArray> =
        Array(vocabSize) { DoubleArray(embeddingDim) { rnd.nextGaussian() * INIT_SCALE } }

    /**
     * Liefert das Embedding einer einzelnen Token-ID.
     *
     * @param tokenId Token-ID im Bereich `0 until vocabSize`.
     * @return Embedding-Vektor der Länge [embeddingDim].
     */
    fun lookup(tokenId: Int): DoubleArray {
        require(tokenId in 0 until vocabSize) { "tokenId $tokenId ausserhalb 0 until $vocabSize" }
        return weights[tokenId]
    }

    /**
     * Liefert die Embeddings einer Token-ID-Sequenz.
     *
     * @param tokenIds Token-IDs.
     * @return Matrix der Form `[tokenIds.size, embeddingDim]`.
     */
    fun lookup(tokenIds: List<Int>): Array<DoubleArray> = Array(tokenIds.size) { lookup(tokenIds[it]) }

    companion object {
        private const val INIT_SCALE = 0.01
    }
}
