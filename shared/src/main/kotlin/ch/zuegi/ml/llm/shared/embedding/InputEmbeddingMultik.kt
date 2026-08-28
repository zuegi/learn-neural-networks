package ch.zuegi.ml.llm.shared.embedding

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get

class InputEmbeddingMultik(
    private val tokenEmbedding: TokenEmbeddingMultik,
    private val positionalEmbedding: PositionalEmbeddingMultik,
) {
    fun forward(tokenIds: List<Int>): NDArray<Double, D2> {
        val tokenVectors = tokenEmbedding.lookup(tokenIds)
        val positionVectors = positionalEmbedding.lookupAll()

        require(tokenVectors.shape[0] == positionVectors.shape[0]) {
            "Sequenzlänge ${tokenVectors.shape[0]} passt nicht zu contextLength ${positionVectors.shape[0]}"
        }
        require(tokenVectors.shape[1] == positionVectors.shape[1]) {
            "embeddingDim ${tokenVectors.shape[1]} passt nicht zu positional embeddingDim ${positionVectors.shape[1]}"
        }

        return mk.ndarray(
            List(tokenVectors.shape[0]) { pos ->
                List(tokenVectors.shape[1]) { dim ->
                    tokenVectors[pos][dim] + positionVectors[pos][dim]
                }
            },
        )
    }
}

