package ch.zuegi.ml.llm

import java.nio.charset.StandardCharsets

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 256

    val model =
        GPTModel(
            vocabSize = tokenizer.vocabSize,
            contextLength = contextLength,
            embeddingDim = embeddingDim,
            numLayers = 2,
            numHeads = 8,
            causal = true,
            seed = 42,
        )

    // Start-Sequenz: erste contextLength Tokens des Textes
    val startIds = tokenIds.take(contextLength)

    // Forward-Pass: Logits der letzten Position
    val logits = model.forward(startIds)
    println("start: ${tokenizer.decode(startIds)}")
    println("logits shape: [${logits.size}, ${logits[0].size}]")

    // Autoregressive Generierung (Modell ist untrainiert -> Ausgabe ist Kauderwelsch)
    val generated = model.generate(startIds, maxNewTokens = 20, greedy = true)
    println("generated ids: $generated")
    println("generated text: ${tokenizer.decode(generated)}")
}

fun readVerdictText(): String {
    val resourcePath = "/text/the-verdict.txt"
    val stream =
        object {}.javaClass.getResourceAsStream(resourcePath)
            ?: error("Ressource nicht gefunden: $resourcePath")

    return stream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
}
