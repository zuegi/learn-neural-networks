package ch.zuegi.ml.llm.kapitel4.demo

import ch.zuegi.ml.llm.kapitel4.GPTConfig
import ch.zuegi.ml.llm.kapitel4.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel4.GenerationConfig
import ch.zuegi.ml.llm.kapitel4.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.ndarray.data.get

fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val generationConfig =
        GenerationConfig(
            maxNewTokens = 10,
            greedy = true,
        )

    val config =
        GPTConfig(
            vocabSize = tokenizer.vocabSize,
            contextLength = 4,
            embeddingDim = 64,
            numLayers = 2,
            numHeads = 8,
            seed = 42L,
        )

    val model = GPTModelMultikTensor(config)

    println("tokenizer vocabSize: ${tokenizer.vocabSize}")
    println("model vocabSize: ${tokenizer.vocabSize}")

    val startIds = tokenIds.take(config.contextLength)
    println("startIds size: ${startIds.size}")
    println("startIds text: ${tokenizer.decode(startIds)}")

    val logits = model.forward(startIds, training = false)
    println("logits flat size: ${logits.size}")
    println("logits shape: [${config.contextLength}, ${config.vocabSize}]")

    val row0 = logits.row(0, config.vocabSize)
    println("erste Logit-Zeile (5 Werte): ${(0 until 5).map { row0.data[it] }}")

    val generated = model.generate(startIds, generationConfig)
    println("generated ids range: min=${generated.minOrNull()} max=${generated.maxOrNull()}")
    println("generated ids: $generated")
    println("generated text: ${tokenizer.decode(generated)}")
}
