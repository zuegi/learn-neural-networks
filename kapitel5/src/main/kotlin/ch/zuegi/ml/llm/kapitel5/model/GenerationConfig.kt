package ch.zuegi.ml.llm.kapitel5.model

data class GenerationConfig(
    val maxNewTokens: Int = 10,
    val temperature: Double = 0.8,
    val topK: Int = 5,
    val greedy: Boolean = false,
    val generatorSeed: Long? = 123L,
) {
    init {
        require(maxNewTokens >= 0) { "maxNewTokens muss >= 0 sein" }
        require(temperature > 0.0) { "temperature muss > 0 sein" }
        require(topK >= 0) { "topK muss >= 0 sein" }
    }
}
