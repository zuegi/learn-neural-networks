package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.scratch.PositionalEmbedding
import ch.zuegi.ml.llm.kapitel2.scratch.SimpleTokenizerV1
import ch.zuegi.ml.llm.kapitel2.scratch.TokenEmbedding
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 256

    val loader = TextDataLoader(tokenIds, contextLength = contextLength, stride = 4, batchSize = 8)

    val tokenEmbedding =
        TokenEmbedding(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = embeddingDim,
            seed = 42,
        )
    val positionalEmbedding =
        PositionalEmbedding(
            contextLength = contextLength,
            embeddingDim = embeddingDim,
            seed = 42,
        )

    val firstSample = loader.samples().first()

    // Token-Embeddings: [contextLength, embeddingDim]
    val tokenVectors = tokenEmbedding.lookup(firstSample.inputIds)

    // Positional-Embeddings: [contextLength, embeddingDim]
    val positionVectors = positionalEmbedding.lookupAll()

    // Input-Embeddings = token + positional (elementweise)
    val inputEmbeddings =
        Array(contextLength) { pos ->
            DoubleArray(embeddingDim) { dim ->
                tokenVectors[pos][dim] + positionVectors[pos][dim]
            }
        }

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [${inputEmbeddings.size}, ${inputEmbeddings[0].size}]")
    println("erstes Input-Embedding (5 Werte): ${inputEmbeddings[0].take(5)}")
}
