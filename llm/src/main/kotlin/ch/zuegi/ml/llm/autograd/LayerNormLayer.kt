package ch.zuegi.ml.llm.autograd

/**
 * Trainierbares Layer-Normalization-Modul auf Basis von [Tensor].
 *
 * Anders als die forward-only `LayerNorm` (die mit `Array<DoubleArray>` arbeitet)
 * haelt dieses Modul [gamma] und [beta] als lernbare [Tensor]-Parameter und
 * nutzt die autograd-faehige Operation [Tensor.layerNorm]. Dadurch koennen die
 * Parameter ueber einen Optimizer wie [SGD] trainiert werden.
 *
 * Normalisiert eine einzelne Token-Zeile (Vektor der Laenge [embeddingDim]).
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param eps kleiner Wert fuer numerische Stabilitaet.
 */
class LayerNormLayer(
    private val embeddingDim: Int,
    private val eps: Double = 1e-5,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    /** Skalierungsparameter, initial 1.0. */
    val gamma: Tensor = Tensor(DoubleArray(embeddingDim) { 1.0 })

    /** Verschiebungsparameter, initial 0.0. */
    val beta: Tensor = Tensor(DoubleArray(embeddingDim) { 0.0 })

    /**
     * Normalisiert eine Token-Zeile.
     *
     * @param x Eingabe-Tensor der Laenge [embeddingDim].
     * @return normalisierter Tensor gleicher Laenge.
     */
    fun forward(x: Tensor): Tensor = x.layerNorm(gamma, beta, eps)

    /**
     * Lernbare Parameter dieses Moduls fuer den Optimizer.
     */
    fun parameters(): List<Tensor> = listOf(gamma, beta)
}
