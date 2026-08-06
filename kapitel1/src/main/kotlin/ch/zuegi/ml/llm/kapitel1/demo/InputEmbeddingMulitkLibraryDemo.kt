package ch.zuegi.ml.llm.kapitel1.demo

import ch.zuegi.ml.llm.kapitel1.library.InputEmbeddingMultik
import ch.zuegi.ml.llm.kapitel1.library.PositionalEmbeddingMultik
import ch.zuegi.ml.llm.kapitel1.library.TokenEmbeddingMultik
import ch.zuegi.ml.llm.kapitel1.scratch.SimpleTokenizerV1
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 256

    val loader = TextDataLoader(tokenIds, contextLength = contextLength, stride = 4, batchSize = 8)

    val tokenEmbedding =
        TokenEmbeddingMultik(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = embeddingDim,
            seed = 42,
        )
    val positionalEmbedding =
        PositionalEmbeddingMultik(
            contextLength = contextLength,
            embeddingDim = embeddingDim,
            seed = 42,
        )
    val inputEmbedding = InputEmbeddingMultik(tokenEmbedding, positionalEmbedding)

    val firstSample = loader.samples().first()
    val inputEmbeddings = inputEmbedding.forward(firstSample.inputIds)

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [${inputEmbeddings.shape[0]}, ${inputEmbeddings.shape[1]}]")
    println("erstes Input-Embedding (5 Werte): ${inputEmbeddings[0].toList().take(5)}")
}
