package ch.zuegi.ml.llm

import ch.zuegi.ml.llm.kapitel2.scratch.SelfAttention
import java.util.Random

/**
 * Multi-Head Self-Attention mit Output-Projektion (nur Forward-Pass).
 *
 * Fuehrt mehrere unabhaengige SelfAttention-Koepfe parallel aus, konkateniert
 * deren Outputs und projiziert das Ergebnis via Wo zurueck auf embeddingDim.
 *
 * Formen:
 *
 *     input        [contextLength, embeddingDim]
 *     concat       [contextLength, numHeads * dK]
 *     output = concat * Wo  -> [contextLength, embeddingDim]
 *
 * Durch die Rueckprojektion auf embeddingDim sind spaeter Residual-Verbindungen
 * (input + output) moeglich.
 *
 * @param embeddingDim Laenge der Input-Embeddings.
 * @param numHeads Anzahl paralleler Attention-Koepfe.
 * @param dK Dimension pro Kopf.
 * @param causal wenn true, maskiert jeder Kopf die Zukunft.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
class MultiHeadAttention(
    private val embeddingDim: Int,
    private val numHeads: Int,
    private val dK: Int,
    causal: Boolean = false,
    seed: Long? = null,
) {
    init {
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(numHeads > 0) { "numHeads muss > 0 sein" }
        require(dK > 0) { "dK muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    private val heads: List<SelfAttention> =
        (0 until numHeads).map { headIndex ->
            SelfAttention(
                embeddingDim = embeddingDim,
                dK = dK,
                causal = causal,
                seed = seed?.let { it + headIndex },
            )
        }

    /**
     * Output-Projektion Wo der Form [numHeads * dK, embeddingDim].
     */
    val wOutput: Array<DoubleArray> =
        Array(numHeads * dK) { DoubleArray(embeddingDim) { rnd.nextGaussian() * INIT_SCALE } }

    /**
     * Forward-Pass ueber alle Koepfe inklusive Output-Projektion.
     *
     * @param input Embeddings der Form [contextLength, embeddingDim].
     * @return projizierter Output [contextLength, embeddingDim].
     */
    fun forward(input: Array<DoubleArray>): Array<DoubleArray> {
        val concatenated = concatHeads(input)
        return matMul(concatenated, wOutput)
    }

    internal fun concatHeads(input: Array<DoubleArray>): Array<DoubleArray> {
        val headOutputs = heads.map { it.forward(input) }

        val contextLength = input.size
        val concatDim = numHeads * dK

        return Array(contextLength) { pos ->
            val row = DoubleArray(concatDim)
            var offset = 0
            for (headOutput in headOutputs) {
                val headRow = headOutput[pos]
                for (d in headRow.indices) {
                    row[offset + d] = headRow[d]
                }
                offset += dK
            }
            row
        }
    }

    private fun matMul(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val rows = a.size
        val inner = b.size
        val cols = b[0].size

        return Array(rows) { i ->
            DoubleArray(cols) { j ->
                var sum = 0.0
                for (k in 0 until inner) {
                    sum += a[i][k] * b[k][j]
                }
                sum
            }
        }
    }

    companion object {
        private const val INIT_SCALE = 0.02
    }
}
