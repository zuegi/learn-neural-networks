package ch.zuegi.ml.llm

data class TrainingSample(
    val inputIds: List<Int>,
    val targetIds: List<Int>,
)
