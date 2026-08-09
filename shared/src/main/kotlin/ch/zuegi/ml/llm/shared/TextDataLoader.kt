package ch.zuegi.ml.llm.shared

/**
 * Erzeugt Trainingsfenster für Next-Token-Prediction aus einer Token-ID-Sequenz.
 *
 * Für jedes Fenster gilt:
 * - inputIds  = tokenIds[start .. start + contextLength - 1]
 * - targetIds = tokenIds[start + 1 .. start + contextLength]
 *
 * Das Fenster wird mit [stride] weitergeschoben.
 *
 * In TextDataLoader bedeutet contextLength:
 * wie viele Token pro inputIds im Sample stehen
 * targetIds ist dieselbe Länge, nur um 1 nach rechts verschoben
 * daraus lernt Modell: “nächstes Token je Position”
 *
 * Beispiel (contextLength = 4):
 * Tokenstream: [10, 11, 12, 13, 14, 15]
 * Sample 1
 *      inputIds = [10, 11, 12, 13]
 *      targetIds = [11, 12, 13, 14]
 * Sample 2
 *      inputIds = [11, 12, 13, 14]
 *      targetIds = [12, 13, 14, 15]
 */
class TextDataLoader(
    private val tokenIds: List<Int>,
    private val contextLength: Int,
    private val stride: Int = 1,
    private val batchSize: Int = 1,
) {
    init {
        require(contextLength > 0) { "contextLength muss > 0 sein" }
        require(stride > 0) { "stride muss > 0 sein" }
        require(batchSize > 0) { "batchSize muss > 0 sein" }
        require(tokenIds.size > contextLength) {
            "tokenIds muss mehr Elemente als contextLength enthalten"
        }
    }

    /**
     * Baut alle Sliding-Window-Samples in Sequenzreihenfolge auf.
     */
    fun samples(): List<TrainingSample> {
        val lastStart = tokenIds.size - contextLength - 1

        return (0..lastStart step stride).map { start ->
            TrainingSample(
                inputIds = tokenIds.subList(start, start + contextLength).toList(),
                targetIds = tokenIds.subList(start + 1, start + contextLength + 1).toList(),
            )
        }
    }

    /**
     * Gruppiert Samples in Mini-Batches mit maximal [batchSize] Elementen.
     * Der letzte Batch kann kleiner sein.
     */
    fun batches(): List<List<TrainingSample>> = samples().chunked(batchSize)

    /**
     * Anzahl erzeugbarer Fenster bei gegebener Sequenzlänge, [contextLength] und [stride].
     */
    fun size(): Int {
        val lastStart = tokenIds.size - contextLength - 1
        return (0..lastStart step stride).count()
    }
}
