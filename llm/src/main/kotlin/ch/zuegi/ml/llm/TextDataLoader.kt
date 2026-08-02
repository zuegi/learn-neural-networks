package ch.zuegi.ml.llm

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

    fun samples(): List<TrainingSample> {
        val lastStart = tokenIds.size - contextLength - 1

        return (0..lastStart step stride).map { start ->
            TrainingSample(
                inputIds = tokenIds.subList(start, start + contextLength).toList(),
                targetIds = tokenIds.subList(start + 1, start + contextLength + 1).toList(),
            )
        }
    }

    fun batches(): List<List<TrainingSample>> = samples().chunked(batchSize)

    fun size(): Int {
        val lastStart = tokenIds.size - contextLength - 1
        return (0..lastStart step stride).count()
    }
}
