package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import java.util.Random

/**
 * Trainierbares position-weises Feed-Forward-Modul auf Basis von [Tensor].
 *
 * Verarbeitet eine Token-Zeile durch zwei lineare Schichten mit GELU
 * dazwischen; die versteckte Dimension ist standardmaessig 4 * embeddingDim.
 *
 *     h   = gelu(W1 · x + b1)   // W1: [hiddenDim, embeddingDim]
 *     out = W2 · h + b2         // W2: [embeddingDim, hiddenDim]
 *
 * Gewichte und Biases sind lernbare [Tensor]-Parameter und werden ueber
 * [parameters] fuer den Optimizer bereitgestellt.
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param hiddenDim versteckte Dimension, Standard 4 * embeddingDim.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class FeedForward(
    private val embeddingDim: Int,
    private val hiddenDim: Int = 4 * embeddingDim,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    // W1: [hiddenDim, embeddingDim], flach row-major
    val w1: Tensor = Tensor(DoubleArray(hiddenDim * embeddingDim) { rnd.nextGaussian() * INIT_SCALE })
    val b1: Tensor = Tensor(DoubleArray(hiddenDim))

    // W2: [embeddingDim, hiddenDim], flach row-major
    val w2: Tensor = Tensor(DoubleArray(embeddingDim * hiddenDim) { rnd.nextGaussian() * INIT_SCALE })
    val b2: Tensor = Tensor(DoubleArray(embeddingDim))

    /**
     * Forward-Pass fuer eine Token-Zeile.
     *
     * @param x Eingabe-Tensor der Laenge [embeddingDim].
     * @return Output-Tensor der Laenge [embeddingDim].
     */
    fun forward(x: Tensor): Tensor {
        val hidden = (x.matVecMul(w1, m = hiddenDim, n = embeddingDim) + b1).gelu()
        return hidden.matVecMul(w2, m = embeddingDim, n = hiddenDim) + b2
    }

    /**
     * Lernbare Parameter dieses Moduls fuer den Optimizer.
     */
    fun parameters(): List<Tensor> = listOf(w1, b1, w2, b2)

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
