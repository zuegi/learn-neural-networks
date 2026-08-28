package ch.zuegi.ml.llm.shared.embedding

/**
 * Kombiniert Token- und Positional-Embeddings zu Input-Embeddings.
 */
class InputEmbedding(
    private val tokenEmbedding: TokenEmbedding,
    private val positionalEmbedding: PositionalEmbedding,
) {
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

