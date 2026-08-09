package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set

/**
 * Einfacher Stochastic-Gradient-Descent-Optimizer.
 *
 * Haelt eine Liste lernbarer Parameter-Tensoren und passt sie nach jedem
 * Backward-Schritt an:
 *
 *     param.data -= learningRate * param.grad
 *
 * Da Gradienten in den Backward-Regeln akkumuliert werden, muessen sie vor
 * jedem neuen Forward/Backward-Schritt via [zeroGrad] zurueckgesetzt werden.
 *
 * @param parameters lernbare Parameter (z.B. Gewichtsmatrizen).
 * @param learningRate Schrittweite des Gradientenabstiegs.
 */
class SGDTensorMultik(
    private val parameters: List<TensorMultik>,
    private val learningRate: Double,
) {
    init {
        require(learningRate > 0.0) { "learningRate muss > 0 sein" }
    }

    /**
     * Aktualisiert alle Parameter einen Schritt entgegen ihrem Gradienten.
     */
    fun step() {
        for (param in parameters) {
            for (i in param.data.indices) {
                param.data[i] -= learningRate * param.grad[i]
            }
        }
    }

    /**
     * Setzt die Gradienten aller Parameter auf 0.
     */
    fun zeroGrad() {
        parameters.forEach { it.zeroGrad() }
    }
}
