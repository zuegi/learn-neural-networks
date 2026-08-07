package ch.zuegi.ml.llm.kapitel3.scratch.embedding

import java.util.Random

/**
 * Lernbare Positional-Embedding-Tabelle.
 *
 * Ordnet jeder Position in der Sequenz einen dichten Vektor der Länge [embeddingDim] zu.
 * Die Gewichte bilden eine Matrix der Form `[contextLength, embeddingDim]`,
 * bei der jede Zeile das Embedding einer Position ist.
 *
 * Positional-Embeddings geben dem Modell Information über die Reihenfolge der Tokens
 * und werden elementweise auf die Token-Embeddings addiert.
 *
 * @param contextLength Anzahl Positionen pro Sequenz.
 * @param embeddingDim Länge eines Embedding-Vektors.
 * @param seed optionaler Seed für reproduzierbare Initialisierung.
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

    /**
     * Embedding-Matrix `[contextLength, embeddingDim]` mit kleinen Zufallswerten.
     */
    val weights: Array<DoubleArray> =
        Array(contextLength) { DoubleArray(embeddingDim) { rnd.nextGaussian() * INIT_SCALE } }

    /**
     * Liefert das Embedding einer einzelnen Position.
     *
     * @param position Position im Bereich `0 until contextLength`.
     * @return Embedding-Vektor der Länge [embeddingDim].
     */
    fun lookup(position: Int): DoubleArray {
        require(position in 0 until contextLength) { "position $position ausserhalb 0 until $contextLength" }
        return weights[position]
    }

    /**
     * Liefert die Embeddings aller Positionen `0 until contextLength`.
     *
     * @return Matrix der Form `[contextLength, embeddingDim]`.
     */
    fun lookupAll(): Array<DoubleArray> = weights

    companion object {
        private const val INIT_SCALE = 0.01
    }
}
