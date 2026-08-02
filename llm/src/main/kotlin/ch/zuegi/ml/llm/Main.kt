package ch.zuegi.ml.llm

import java.nio.charset.StandardCharsets

// Wörter + interne Apostrophe/Bindestriche, Satzzeichen separat als Tokens

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val loader = TextDataLoader(tokenIds, contextLength = 4, stride = 4, batchSize = 8)

    println(loader.size())
    println(loader.samples().first().inputIds)
    println(loader.samples().first().targetIds)
}

fun readVerdictText(): String {
    val resourcePath = "/text/the-verdict.txt"
    val stream =
        object {}.javaClass.getResourceAsStream(resourcePath)
            ?: error("Ressource nicht gefunden: $resourcePath")

    return stream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
}
