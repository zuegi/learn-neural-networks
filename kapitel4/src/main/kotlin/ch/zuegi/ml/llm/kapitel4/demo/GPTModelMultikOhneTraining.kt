package ch.zuegi.ml.llm.kapitel4.demo

import ch.zuegi.ml.llm.kapitel4.GPTConfig
import ch.zuegi.ml.llm.kapitel4.GPTModelMultik
import ch.zuegi.ml.llm.kapitel4.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val config =
        GPTConfig(
            vocabSize = tokenizer.vocabSize,
            contextLength = 4,
            embeddingDim = 64,
            numLayers = 2,
            numHeads = 8,
            seed = 42L,
        )

    val model = GPTModelMultik(config)

    println("tokenizer vocabSize: ${tokenizer.vocabSize}")
    println("model vocabSize: ${tokenizer.vocabSize}")

    val startIds = tokenIds.take(config.contextLength)
    println("startIds size: ${startIds.size}")
    println("startIds text: ${tokenizer.decode(startIds)}")

    val logits = model.forward(startIds, training = false)
    println("logits shape: [${logits.shape[0]}, ${logits.shape[1]}]")
    println("logits flat size: ${logits.size}")

    val row0 = logits[0]
    println("erste Logit-Zeile (5 Werte): ${row0.toList().take(5)}")

    val generated = model.generate(startIds, maxNewTokens = 10, greedy = true)
    println("generated ids range: min=${generated.minOrNull()} max=${generated.maxOrNull()}")
    println("generated ids: $generated")
    println("generated text: ${tokenizer.decode(generated)}")
}
