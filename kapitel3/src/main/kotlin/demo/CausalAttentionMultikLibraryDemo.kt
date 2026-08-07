package demo

import ch.zuegi.ml.llm.kapitel3.library.embedding.InputEmbeddingMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.PositionalEmbeddingMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.TokenEmbeddingMultik
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import library.CausalAttentionMultik
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val registry = Encodings.newDefaultEncodingRegistry()
    val tokenizer = registry.getEncoding(EncodingType.R50K_BASE)

    val rawText = readVerdictText()
    val tokenIds = tokenizer.encode(rawText).toArray().toList()

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
        CausalAttentionMultik(
            embeddingDim = embeddingDim,
            dK = 64,
            dropoutProb = 0.1,
            seed = 42,
        )

    val contextTrain = attention.forward(embeddings, training = true)
    val contextEval = attention.forward(embeddings, training = false)

    println("inputIds: ${firstSample.inputIds}")
    println("shape inputEmbeddings: [${embeddings.shape[0]}, ${embeddings.shape[1]}]")
    println("erstes Input-Embedding (5 Werte): ${embeddings[0].toList().take(5)}")
    println("causal attention output (train) shape: [${contextTrain.shape[0]}, ${contextTrain.shape[1]}]")
    println("causal attention output (eval) shape: [${contextEval.shape[0]}, ${contextEval.shape[1]}]")
}
