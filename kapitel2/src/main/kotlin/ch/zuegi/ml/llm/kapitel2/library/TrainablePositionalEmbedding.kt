package ch.zuegi.ml.llm.kapitel2.library

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

/**
 * Trainierbare Positional-Embedding-Tabelle.
 *
 * Ordnet jeder Position in einer Sequenz einen dichten Vektor der Länge [embeddingDim] zu.
 * Die Gewichte bilden eine Matrix der Form `[contextLength, embeddingDim]`,
 * bei der jede Zeile das Embedding einer Position ist.
 *
 * Anders als bei festen Positionsencodings sind diese Vektoren trainierbar:
 * Sie werden im Forward-Pass gelesen, im Backward-Pass mit Gradienten versehen
 * und im Optimizer-Schritt aktualisiert.
 *
 * @param contextLength maximale Sequenzlänge, für die Positionsvektoren bereitgestellt werden.
 * @param embeddingDim Länge eines einzelnen Positionsvektors.
 * @param seed optionaler Seed für reproduzierbare Initialisierung.
 */
class TrainablePositionalEmbedding(
    val contextLength: Int,
    val embeddingDim: Int,
    seed: Long? = null,
) {
    init {
        require(contextLength > 0) { "contextLength muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()
    private val initScale = 1.0 / sqrt(embeddingDim.toDouble())

    /**
     * Trainierbare Gewichtsmatrix `[contextLength, embeddingDim]`.
     */
    val weights: D2Array<Double> =
        run {
            val flat = DoubleArray(contextLength * embeddingDim) { rnd.nextGaussian() * initScale }
            mk.ndarray(flat, contextLength, embeddingDim)
        }

    private var grad: D2Array<Double> = mk.zeros(contextLength, embeddingDim)
    private var lastPositions: List<Int> = emptyList()

    /**
     * Liefert das Embedding einer einzelnen Position.
     *
     * @param position Position im Bereich `0 until contextLength`.
     * @return Positionsvektor der Länge [embeddingDim].
     */
    operator fun get(position: Int): MultiArray<Double, D1> {
        require(position in 0 until contextLength) {
            "position $position ausserhalb 0 until $contextLength"
        }
        return weights[position]
    }

    /**
     * Forward-Pass für eine Sequenz gegebener Länge.
     *
     * Intern werden die Positionen `0 until sequenceLength` verwendet.
     * Die zugehörigen Positionsvektoren werden als Matrix der Form
     * `[sequenceLength, embeddingDim]` zurückgegeben.
     *
     * @param sequenceLength Länge der aktuellen Sequenz.
     * @return Matrix mit Positionsvektoren für alle Positionen der Sequenz.
     */
    @Suppress("ktlint:standard:no-consecutive-comments")
    // tag::trainablePositionalForward[]
    fun forward(sequenceLength: Int): D2Array<Double> {
        require(sequenceLength in 1..contextLength) {
            "sequenceLength $sequenceLength muss im Bereich 1..$contextLength liegen"
        }
        lastPositions = (0 until sequenceLength).toList()
        return mk.stack(lastPositions.map { weights[it] })
    }
    // end::trainablePositionalForward[]

    /**
     * Backward-Pass: akkumuliert Gradienten auf den im Forward-Pass verwendeten Positionen.
     *
     * @param gradOutput Gradient bezüglich der Forward-Ausgabe mit Form `[seqLen, embeddingDim]`.
     */
    @Suppress("ktlint:standard:no-consecutive-comments")
    // tag::trainablePositionalBackward[]
    fun backward(gradOutput: D2Array<Double>) {
        require(gradOutput.shape[0] == lastPositions.size) {
            "gradOutput hat ${gradOutput.shape[0]} Zeilen, erwartet wurden ${lastPositions.size}"
        }
        require(gradOutput.shape[1] == embeddingDim) {
            "gradOutput hat Dimension ${gradOutput.shape[1]}, erwartet wurde $embeddingDim"
        }

        for ((i, position) in lastPositions.withIndex()) {
            for (d in 0 until embeddingDim) {
                grad[position, d] = grad[position, d] + gradOutput[i, d]
            }
        }
    }
    // end::trainablePositionalBackward[]

    /**
     * Führt ein SGD-Update auf allen Positionszeilen durch und setzt danach die Gradienten zurück.
     *
     * @param learningRate Lernrate für den Update-Schritt.
     */
    @Suppress("ktlint:standard:no-consecutive-comments")
    // tag::trainablePositionalStep[]
    fun step(learningRate: Double) {
        for (position in 0 until contextLength) {
            for (d in 0 until embeddingDim) {
                weights[position, d] = weights[position, d] - learningRate * grad[position, d]
            }
        }
        zeroGrad()
    }
    // end::trainablePositionalStep[]

    // tag::trainablePositionalZeroGrad[]

    /**
     * Setzt den akkumulierten Gradienten auf Null zurück.
     */
    fun zeroGrad() {
        grad = mk.zeros(contextLength, embeddingDim)
    }
    // end::trainablePositionalZeroGrad[]
}

