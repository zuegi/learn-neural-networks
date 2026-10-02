package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import java.util.Random

/**
 * Trainierbare Lookup-Tabelle `[numEmbeddings, embeddingDim]` mit Autograd.
 *
 * Token- und Positions-Embedding sind dieselbe Operation: Zeile `index` lesen.
 * Der Gradient fliesst nur in die gelesene Zeile zurück.
 */
class EmbeddingMultikTensor(
    private val numEmbeddings: Int,
    private val embeddingDim: Int,
    initScale: Double,
    rnd: Random,
) {
    init {
        require(numEmbeddings > 0) { "numEmbeddings muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    val weight: TensorMultik =
        TensorMultik(mk.ndarray(DoubleArray(numEmbeddings * embeddingDim) { rnd.nextGaussian() * initScale }))

    // tag::embedding-forward[]
    fun forward(index: Int): TensorMultik {
        require(index in 0 until numEmbeddings) { "index $index ausserhalb 0 until $numEmbeddings" }
        return weight.row(index, embeddingDim)
    }
    // end::embedding-forward[]

    fun parameters(): List<TensorMultik> = listOf(weight)
}
