package ch.zuegi.ml.llm.shared

data class TrainingSample(
    val inputIds: List<Int>,
    val targetIds: List<Int>,
)
