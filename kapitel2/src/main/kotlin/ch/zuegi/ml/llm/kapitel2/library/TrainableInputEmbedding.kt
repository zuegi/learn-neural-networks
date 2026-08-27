package ch.zuegi.ml.llm.kapitel2.library

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get

/**
 * Kombiniert trainierbare Token- und Positional-Embeddings zu trainierbaren Input-Embeddings.
 *
 * Die Forward-Ausgabe entsteht durch elementweise Addition von
 * `TrainableTokenEmbedding.forward(tokenIds)` und
 * `TrainablePositionalEmbedding.forward(tokenIds.size)`.
 *
 * Im Backward-Pass wird derselbe Gradient an beide Teilkomponenten weitergegeben,
 * da für `input = token + position` gilt:
 * - dL/dtoken = dL/dinput
 * - dL/dposition = dL/dinput
 *
 * @param tokenEmbedding trainierbare Token-Embedding-Tabelle.
 * @param positionalEmbedding trainierbare Positional-Embedding-Tabelle.
 */
class TrainableInputEmbedding(
    private val tokenEmbedding: TrainableTokenEmbedding,
    private val positionalEmbedding: TrainablePositionalEmbedding,
) {
    private var lastSequenceLength: Int = 0
    private var lastEmbeddingDim: Int = 0

    /**
     * Erzeugt Input-Embeddings für eine Token-ID-Sequenz.
     *
     * @param tokenIds Token-IDs der aktuellen Sequenz.
     * @return Matrix der Form `[tokenIds.size, embeddingDim]`.
     */
    // tag::trainableInputForward[]
    @Suppress("ktlint:standard:no-consecutive-comments")
    fun forward(tokenIds: List<Int>): D2Array<Double> {
        val tokenVectors = tokenEmbedding.forward(tokenIds)
        val positionVectors = positionalEmbedding.forward(tokenIds.size)

        require(tokenVectors.shape[0] == positionVectors.shape[0]) {
            "Sequenzlänge ${tokenVectors.shape[0]} passt nicht zu contextLength ${positionVectors.shape[0]}"
        }
        require(tokenVectors.shape[1] == positionVectors.shape[1]) {
            "embeddingDim ${tokenVectors.shape[1]} passt nicht zu positional embeddingDim ${positionVectors.shape[1]}"
        }

        lastSequenceLength = tokenVectors.shape[0]
        lastEmbeddingDim = tokenVectors.shape[1]

        return mk.ndarray(
            List(tokenVectors.shape[0]) { pos ->
                List(tokenVectors.shape[1]) { dim ->
                    tokenVectors[pos][dim] + positionVectors[pos][dim]
                }
            },
        )
    }
    // end::trainableInputForward[]

    /**
     * Leitet den Gradienten an Token- und Positional-Embedding weiter.
     *
     * @param gradOutput Gradient bezüglich der Input-Embedding-Ausgabe.
     */
    // tag::trainableInputBackward[]
    @Suppress("ktlint:standard:no-consecutive-comments")
    fun backward(gradOutput: D2Array<Double>) {
        require(lastSequenceLength > 0) {
            "forward() muss vor backward() aufgerufen werden"
        }
        require(gradOutput.shape[0] == lastSequenceLength) {
            "gradOutput hat ${gradOutput.shape[0]} Zeilen, erwartet wurden $lastSequenceLength"
        }
        require(gradOutput.shape[1] == lastEmbeddingDim) {
            "gradOutput hat Dimension ${gradOutput.shape[1]}, erwartet wurde $lastEmbeddingDim"
        }

        tokenEmbedding.backward(gradOutput)
        positionalEmbedding.backward(gradOutput)
    }
    // end::trainableInputBackward[]

    /**
     * Führt den Optimizer-Schritt für beide Teilkomponenten aus.
     *
     * @param learningRate Lernrate für den SGD-Schritt.
     */
    // tag::trainableInputStep[]
    @Suppress("ktlint:standard:no-consecutive-comments")
    fun step(learningRate: Double) {
        tokenEmbedding.step(learningRate)
        positionalEmbedding.step(learningRate)
    }
    // end::trainableInputStep[]

    /**
     * Setzt die Gradienten beider Teilkomponenten auf Null zurück.
     */
    // tag::trainableInputZeroGrad[]
    @Suppress("ktlint:standard:no-consecutive-comments")
    fun zeroGrad() {
        tokenEmbedding.zeroGrad()
        positionalEmbedding.zeroGrad()
    }
    // end::trainableInputZeroGrad[]
}
