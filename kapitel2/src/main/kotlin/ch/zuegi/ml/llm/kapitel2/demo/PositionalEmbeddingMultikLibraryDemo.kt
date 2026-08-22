package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.library.PositionalEmbeddingMultik
import ch.zuegi.ml.llm.kapitel2.library.R50kBpeTokenizer
import ch.zuegi.ml.llm.kapitel2.library.TokenEmbeddingMultik
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val rawText = readVerdictText()
    val tokenizer = R50kBpeTokenizer()
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

    val firstSample = loader.samples().first()

    // Token-Embeddings: [contextLength, embeddingDim]
    val tokenVectors = tokenEmbedding.lookup(firstSample.inputIds)

    // Positional-Embeddings: [contextLength, embeddingDim]
    val positionVectors = positionalEmbedding.lookupAll()

    // Input-Embeddings = token + positional (elementweise)
    val inputEmbeddings =
        mk.ndarray(
            List(contextLength) { pos ->
                List(embeddingDim) { dim ->
                    tokenVectors[pos][dim] + positionVectors[pos][dim]
                }
            },
        )

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [${inputEmbeddings.shape[0]}, ${inputEmbeddings.shape[1]}]")
    println("erstes Input-Embedding (5 Werte): ${inputEmbeddings[0].toList().take(5)}")
}
