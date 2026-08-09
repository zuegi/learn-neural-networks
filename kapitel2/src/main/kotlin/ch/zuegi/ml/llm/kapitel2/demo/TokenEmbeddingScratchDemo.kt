package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.scratch.SimpleTokenizerV1
import ch.zuegi.ml.llm.kapitel2.scratch.TokenEmbedding
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val loader = TextDataLoader(tokenIds, contextLength = 4, stride = 4, batchSize = 8)

    val embedding =
        TokenEmbedding(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = 256,
            seed = 42,
        )

    val firstSample = loader.samples().first()
    val inputEmbeddings = embedding.lookup(firstSample.inputIds)

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [${inputEmbeddings.size}, ${inputEmbeddings[0].size}]")
    println("erstes Token-Embedding (5 Werte): ${inputEmbeddings[0].take(5)}")
}
