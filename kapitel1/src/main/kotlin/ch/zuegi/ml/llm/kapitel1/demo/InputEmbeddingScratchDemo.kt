package ch.zuegi.ml.llm.kapitel1.demo

import ch.zuegi.ml.llm.kapitel1.scratch.InputEmbedding
import ch.zuegi.ml.llm.kapitel1.scratch.PositionalEmbedding
import ch.zuegi.ml.llm.kapitel1.scratch.SimpleTokenizerV1
import ch.zuegi.ml.llm.kapitel1.scratch.TokenEmbedding
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 256

    val loader = TextDataLoader(tokenIds, contextLength = contextLength, stride = 4, batchSize = 8)

    val inputEmbedding =
        InputEmbedding(
            tokenEmbedding = TokenEmbedding(tokenizer.vocabSize, embeddingDim, seed = 42),
            positionalEmbedding = PositionalEmbedding(contextLength, embeddingDim, seed = 42),
        )

    val firstSample = loader.samples().first()
    val embeddings = inputEmbedding.forward(firstSample.inputIds)

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [${embeddings.size}, ${embeddings[0].size}]")
    println("erstes Input-Embedding (5 Werte): ${embeddings[0].take(5)}")
}
