package ch.zuegi.ml.llm

/**
 * Kombiniert Token- und Positional-Embeddings zu Input-Embeddings.
 *
 * Für eine Token-ID-Sequenz werden die zugehörigen Token-Embeddings mit den
 * Positional-Embeddings der jeweiligen Position elementweise addiert.
 *
 * Ergebnisform: `[contextLength, embeddingDim]`.
 *
 * @param tokenEmbedding lernbare Token-Embedding-Tabelle.
 * @param positionalEmbedding lernbare Positional-Embedding-Tabelle.
 */
class InputEmbedding(
    private val tokenEmbedding: TokenEmbedding,
    private val positionalEmbedding: PositionalEmbedding,
) {
    /**
     * Erzeugt die Input-Embeddings für eine Token-ID-Sequenz.
     *
     * Die Länge von [tokenIds] muss der `contextLength` der
     * [positionalEmbedding]-Tabelle entsprechen.
     *
     * @param tokenIds Token-IDs der Sequenz.
     * @return Matrix `[tokenIds.size, embeddingDim]` aus token + positional.
     */
    fun forward(tokenIds: List<Int>): Array<DoubleArray> {
        val tokenVectors = tokenEmbedding.lookup(tokenIds)
        val positionVectors = positionalEmbedding.lookupAll()

        require(tokenVectors.size == positionVectors.size) {
            "Sequenzlänge ${tokenVectors.size} passt nicht zu contextLength ${positionVectors.size}"
        }

        return Array(tokenVectors.size) { pos ->
            val token = tokenVectors[pos]
            val position = positionVectors[pos]
            DoubleArray(token.size) { dim -> token[dim] + position[dim] }
        }
    }
}
