package ch.zuegi.ml.llm.kapitel3.demo

import ch.zuegi.ml.llm.kapitel3.library.MultiHeadAttentionMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.InputEmbeddingMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.PositionalEmbeddingMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.TokenEmbeddingMultik
import ch.zuegi.ml.llm.kapitel3.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 256
    val vocabSize = 50257

    val loader = TextDataLoader(tokenIds, contextLength = contextLength, stride = 4, batchSize = 8)

    val inputEmbedding =
        InputEmbeddingMultik(
            tokenEmbedding = TokenEmbeddingMultik(vocabSize, embeddingDim, seed = 42),
            positionalEmbedding = PositionalEmbeddingMultik(contextLength, embeddingDim, seed = 42),
        )

    val firstSample = loader.samples().first()
    val embeddings = inputEmbedding.forward(firstSample.inputIds)

    val attention =
        MultiHeadAttentionMultik(
            embeddingDim = embeddingDim,
            numHeads = 8,
            dK = 64,
            dropoutProb = 0.1,
            seed = 42,
        )

    val contextTrain = attention.forward(embeddings, training = true)
    val contextEval = attention.forward(embeddings, training = false)

    println("inputIds: ${firstSample.inputIds}")
    println("shape inputEmbeddings: [${embeddings.shape[0]}, ${embeddings.shape[1]}]")
    println("erstes Input-Embedding (5 Werte): ${embeddings[0].toList().take(5)}")
    println("multi-head output (train) shape: [${contextTrain.shape[0]}, ${contextTrain.shape[1]}]")
    println("multi-head output (eval) shape: [${contextEval.shape[0]}, ${contextEval.shape[1]}]")
}
