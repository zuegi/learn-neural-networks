import ch.zuegi.ml.llm.kapitel5.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.shared.DummyLanguageModel
import ch.zuegi.ml.llm.shared.generateTextSimple

fun main() {
    val tokenizer = GPT2Tokenizer()
    val model = DummyLanguageModel(vocabSize = 50_257)
    val startIds = listOf(15496, 11) // z.B. "Hello,"
    val outIds =
        generateTextSimple(
            model = model,
            startTokenIds = startIds,
            maxNewTokens = 20,
            contextSize = 128,
        )
    println(outIds)
}
