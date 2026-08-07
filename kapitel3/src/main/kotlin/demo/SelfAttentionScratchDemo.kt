package ch.zuegi.ml.llm.kapitel3.demo

import ch.zuegi.ml.llm.kapitel3.scratch.SelfAttention
import ch.zuegi.ml.llm.kapitel3.scratch.embedding.InputEmbedding
import ch.zuegi.ml.llm.kapitel3.scratch.embedding.PositionalEmbedding
import ch.zuegi.ml.llm.kapitel3.scratch.embedding.TokenEmbedding
import ch.zuegi.ml.llm.kapitel3.scratch.tokenize.SimpleTokenizerV1
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
    val attention = SelfAttention(embeddingDim = embeddingDim, dK = 64, seed = 42, causal = true)
    val context = attention.forward(embeddings) // [ctx, 64]

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [${embeddings.size}, ${embeddings[0].size}]")
    println("erstes Input-Embedding (5 Werte): ${embeddings[0].take(5)}")
    println("attention output shape: [${context.size}, ${context[0].size}]")
}
