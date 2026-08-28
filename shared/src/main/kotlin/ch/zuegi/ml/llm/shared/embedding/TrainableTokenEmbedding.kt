package ch.zuegi.ml.llm.shared.embedding

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import java.util.Random
import kotlin.math.sqrt

class TrainableTokenEmbedding(
    val vocabSize: Int,
    val embeddingDim: Int,
    seed: Long? = null,
) {
    init {
        require(vocabSize > 0) { "vocabSize muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()
    private val initScale = 1.0 / sqrt(embeddingDim.toDouble())

    // Trainierbare Gewichte
    var weights: D2Array<Double> =
        run {
            val flat = DoubleArray(vocabSize * embeddingDim) { rnd.nextGaussian() * initScale }
            mk.ndarray(flat, vocabSize, embeddingDim)
        }

    // Gradienten-Akkumulator, gleiche Form wie weights
    private var grad: D2Array<Double> = mk.zeros(vocabSize, embeddingDim)

    // Merkt sich die zuletzt im Forward-Pass genutzten Token-IDs
    private var lastTokenIds: List<Int> = emptyList()

    operator fun get(tokenId: Int): MultiArray<Double, D1> {
        require(tokenId in 0 until vocabSize) { "tokenId $tokenId ausserhalb 0 until $vocabSize" }
        return weights[tokenId]
    }

    /**
     * Forward-Pass: gibt die Embedding-Matrix für die übergebenen Token-IDs zurück
     * und merkt sich die IDs für den nachfolgenden Backward-Pass.
     */
    // tag::tokenEmbeddingForward[]
    @Suppress("ktlint:standard:no-consecutive-comments")
    fun forward(tokenIds: List<Int>): D2Array<Double> {
        tokenIds.forEach {
            require(it in 0 until vocabSize) { "tokenId $it ausserhalb 0 until $vocabSize" }
        }
        lastTokenIds = tokenIds
        return mk.stack(tokenIds.map { weights[it] })
    }
    // end::tokenEmbeddingForward[]

    /**
     * Backward-Pass: nimmt den Gradienten bezüglich der Forward-Ausgabe entgegen
     * (gleiche Form wie forward()-Ergebnis: [seqLen, embeddingDim])
     * und akkumuliert ihn zeilenweise in die passenden Embedding-Zeilen.
     */
    // tag::tokenEmbeddingBackward[]
    @Suppress("ktlint:standard:no-consecutive-comments")
    fun backward(gradOutput: D2Array<Double>) {
        require(gradOutput.shape[0] == lastTokenIds.size) {
            "gradOutput hat ${gradOutput.shape[0]} Zeilen, erwartet wurden ${lastTokenIds.size}"
        }
        require(gradOutput.shape[1] == embeddingDim) {
            "gradOutput hat Dimension ${gradOutput.shape[1]}, erwartet wurde $embeddingDim"
        }

        for ((i, tokenId) in lastTokenIds.withIndex()) {
            for (d in 0 until embeddingDim) {
                grad[tokenId, d] = grad[tokenId, d] + gradOutput[i, d]
            }
        }
    }
    // end::tokenEmbeddingBackward[]

    /**
     * SGD-Update: weights -= lr * grad, danach Gradienten zurücksetzen.
     */
    // tag::tokenEmbeddingStep[]
    @Suppress("ktlint:standard:no-consecutive-comments")
    fun step(learningRate: Double) {
        for (v in 0 until vocabSize) {
            for (d in 0 until embeddingDim) {
                weights[v, d] = weights[v, d] - learningRate * grad[v, d]
            }
        }
        zeroGrad()
    }
    // end::tokenEmbeddingStep[]

    // tag::tokenEmbedingZeroGrad[]
    fun zeroGrad() {
        grad = mk.zeros(vocabSize, embeddingDim)
    }
    // end::tokenEmbedingZeroGrad[]
}
