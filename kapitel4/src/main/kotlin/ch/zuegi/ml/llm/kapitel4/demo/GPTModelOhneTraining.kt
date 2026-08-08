package ch.zuegi.ml.llm.kapitel4.demo

import ch.zuegi.ml.llm.kapitel4.GPTConfig
import ch.zuegi.ml.llm.kapitel4.GPTModel
import ch.zuegi.ml.llm.kapitel4.scratch.tokenize.SimpleTokenizerV1
import ch.zuegi.ml.llm.shared.readVerdictText

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val config =
        GPTConfig(
            vocabSize = tokenizer.vocabSize, // SimpleTokenizerV1 kennt nur IDs aus seinem eigenen Vokabular.
            contextLength = 4,
            embeddingDim = 64,
            numLayers = 2,
            numHeads = 8,
            seed = 42L,
        )
    val model = GPTModel(config)

    println("tokenizer vocabSize: ${tokenizer.vocabSize}")
    println("model vocabSize: ${config.vocabSize}")

    // Start-Sequenz: erste contextLength Tokens des Textes
    val startIds = tokenIds.take(config.contextLength)
    println("startIds size: ${startIds.size}")
    println("startIds text: ${tokenizer.decode(startIds)}")

    // Forward-Pass: Logits der letzten Position
    val logits = model.forward(startIds)
    println("logits shape: [${config.contextLength}, ${config.vocabSize}]")
    println("logits flat size: ${logits.size}")

    // Zeile 0 aus Tensor lesen
    val row0 = logits.row(0, config.vocabSize)
    println("erste Logit-Zeile (5 Werte): ${row0.data.take(5)}")

    // Autoregressive Generierung (Modell ist untrainiert -> Ausgabe ist Kauderwelsch)
    val generated = model.generate(startIds, maxNewTokens = 10, greedy = true)
    println("generated ids range: min=${generated.minOrNull()} max=${generated.maxOrNull()}")
    println("generated ids: $generated")
    println("generated text: ${tokenizer.decode(generated)}")
}
