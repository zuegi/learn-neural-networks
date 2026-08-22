package ch.zuegi.ml.llm.shared

/**
 * Erzeugt Trainingsfenster für Next-Token-Prediction aus einer Token-ID-Sequenz.
 *
 * Für jedes Fenster gilt:
 * - inputIds  = tokenIds[start .. start + contextLength - 1]
 * - targetIds = tokenIds[start + 1 .. start + contextLength]
 *
 * Das Fenster wird mit [stride] weitergeschoben und mit [batchSize] zu Mini-Batches gruppiert.
 *
 * [contextLength] gibt an, wie viele Token pro `inputIds` in einem Sample stehen.
 * `targetIds` hat dieselbe Länge, ist aber um ein Token nach rechts verschoben.
 * Daraus lernt das Modell an jeder Position das jeweils nächste Token.
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
            "tokenIds.size muss größer als contextLength sein"
        }
    }

    /**
     * Baut alle Sliding-Window-Samples in Sequenzreihenfolge auf.
     */
    @Suppress("ktlint:standard:no-consecutive-comments")
    // tag::samples[]
    fun samples(): List<TrainingSample> {
        return windowStarts().map { start ->
            TrainingSample(
                inputIds = tokenIds.subList(start, start + contextLength).toList(),
                targetIds = tokenIds.subList(start + 1, start + contextLength + 1).toList(),
            )
        }
    }
    // end::samples[]

    /**
     * Gruppiert Samples in Mini-Batches mit maximal [batchSize] Elementen.
     * Der letzte Batch kann kleiner sein.
     */
    @Suppress("ktlint:standard:no-consecutive-comments")
    // tag::batches[]
    fun batches(): List<List<TrainingSample>> = samples().chunked(batchSize)
    // end::batches[]

    /**
     * Anzahl erzeugbarer Fenster bei gegebener Sequenzlänge, [contextLength] und [stride].
     */
    @Suppress("ktlint:standard:no-consecutive-comments")
    // tag::size[]
    fun size(): Int = windowStarts().count()
    // end::size[]

    private fun windowStarts(): IntProgression = 0..lastStart() step stride

    private fun lastStart(): Int = tokenIds.size - contextLength - 1
}
