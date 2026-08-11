package ch.zuegi.ml.llm.shared

interface Tokenizer {
    fun encode(text: String): List<Int>

    fun decode(tokenIds: List<Int>): String
}
