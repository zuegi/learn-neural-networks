package ch.zuegi.ml.llm.kapitel3.demo

import ch.zuegi.ml.llm.kapitel3.scratch.MultiHeadAttentionWrapper
import ch.zuegi.ml.llm.shared.embedding.InputEmbedding
import ch.zuegi.ml.llm.shared.embedding.PositionalEmbedding
import ch.zuegi.ml.llm.shared.embedding.TokenEmbedding
import ch.zuegi.ml.llm.shared.tokenize.SimpleTokenizerV1
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

    val attention =
        MultiHeadAttentionWrapper(
            embeddingDim = embeddingDim,
            numHeads = 8,
            dK = 64,
            dropoutProb = 0.1,
            seed = 42,
        )

    val contextTrain = attention.forward(embeddings, training = true)
    val contextEval = attention.forward(embeddings, training = false)

    println("inputIds: ${firstSample.inputIds}")
    println("shape inputEmbeddings: [${embeddings.size}, ${embeddings[0].size}]")
    println("erstes Input-Embedding (5 Werte): ${embeddings[0].take(5)}")
    println("multi-head output (train) shape: [${contextTrain.size}, ${contextTrain[0].size}]")
    println("multi-head output (eval) shape: [${contextEval.size}, ${contextEval[0].size}]")
}
