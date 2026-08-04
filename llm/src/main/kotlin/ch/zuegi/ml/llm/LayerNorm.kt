package ch.zuegi.ml.llm

import kotlin.math.sqrt

/**
 * Layer Normalization (nur Forward-Pass).
 *
 * Normalisiert jede Zeile (jedes Token) eines Inputs [contextLength, embeddingDim]
 * ueber die Feature-Achse auf Mittelwert 0 und Varianz 1. Danach wird mit den
 * lernbaren Parametern gamma (Skalierung) und beta (Verschiebung) transformiert.
 *
 * Pro Zeile:
 *
 *     mean = Durchschnitt der Zeile
 *     var  = Varianz der Zeile
 *     norm = (x - mean) / sqrt(var + eps)
 *     out  = norm * gamma + beta
 *
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param eps kleiner Wert fuer numerische Stabilitaet.
 */
class LayerNorm(
    private val embeddingDim: Int,
    private val eps: Double = 1e-5,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    /**
     * Skalierungsparameter, initial 1.0.
     */
    val gamma: DoubleArray = DoubleArray(embeddingDim) { 1.0 }

    /**
     * Verschiebungsparameter, initial 0.0.
     */
    val beta: DoubleArray = DoubleArray(embeddingDim) { 0.0 }

    /**
     * Normalisiert jede Zeile des Inputs.
     *
     * @param input Form [contextLength, embeddingDim].
     * @return normalisierter Output gleicher Form.
     */
    fun forward(input: Array<DoubleArray>): Array<DoubleArray> = Array(input.size) { i -> normalizeRow(input[i]) }

    private fun normalizeRow(row: DoubleArray): DoubleArray {
        val mean = row.average()
        val variance = row.sumOf { (it - mean) * (it - mean) } / row.size
        val denom = sqrt(variance + eps)

        return DoubleArray(row.size) { j ->
            val normalized = (row[j] - mean) / denom
            normalized * gamma[j] + beta[j]
        }
    }
}
