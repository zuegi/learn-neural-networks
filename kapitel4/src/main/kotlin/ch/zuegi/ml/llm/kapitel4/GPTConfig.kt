package ch.zuegi.ml.llm.kapitel4

/**
 * Konfiguration fuer ein GPT-Modell.
 *
 * @param vocabSize Groesse des Vokabulars.
 * @param contextLength Anzahl Positionen pro Sequenz.
 * @param embeddingDim Laenge eines Token-Vektors.
 * @param numLayers Anzahl gestapelter Transformer-Bloecke.
 * @param numHeads Anzahl Attention-Koepfe pro Block.
 * @param hiddenDim versteckte Dimension der Feed-Forward-Netze, Standard 4 * embeddingDim.
 * @param causal wenn true, maskiert die Attention die Zukunft.
 * @param dropoutProb Attention-Dropout.
 * @param seed optionaler Seed fuer reproduzierbare Initialisierung.
 */
data class GPTConfig(
    val vocabSize: Int = 50257,
    val contextLength: Int = 256,
    val embeddingDim: Int = 256,
    val numLayers: Int = 6,
    val numHeads: Int = 8,
    val hiddenDim: Int = 4 * embeddingDim,
    val causal: Boolean = true,
    val dropoutProb: Double = 0.1,
    val useQkvBias: Boolean = false,
    val useOutputBias: Boolean = false,
    val seed: Long? = null,
) {
    init {
        require(vocabSize > 0) { "vocabSize muss > 0 sein" }
        require(contextLength > 0) { "contextLength muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
        require(numLayers > 0) { "numLayers muss > 0 sein" }
        require(numHeads > 0) { "numHeads muss > 0 sein" }
        require(embeddingDim % numHeads == 0) {
            "embeddingDim ($embeddingDim) muss durch numHeads ($numHeads) teilbar sein"
        }
        require(hiddenDim > 0) { "hiddenDim muss > 0 sein" }
        require(dropoutProb in 0.0..1.0) { "dropoutProb muss in [0.0, 1.0] liegen" }
    }

    /** Dimension pro Attention-Kopf. */
    val dK: Int get() = embeddingDim / numHeads
}
