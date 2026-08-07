package ch.zuegi.ml.llm.kapitel3.demo

import ch.zuegi.ml.llm.kapitel3.library.SelfAttentionMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.InputEmbeddingMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.PositionalEmbeddingMultik
import ch.zuegi.ml.llm.kapitel3.library.embedding.TokenEmbeddingMultik
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    // 1. Registry laden
    val registry = Encodings.newDefaultEncodingRegistry()

    // 2. GPT-2 Encoding (r50k_base) abrufen
    val tokenizer = registry.getEncoding(EncodingType.R50K_BASE)

    // 3. Text codieren
    val rawText = readVerdictText()
    val tokenIds = tokenizer.encode(rawText).toArray().toList()

    val contextLength = 4
    val embeddingDim = 256

    val loader = TextDataLoader(tokenIds, contextLength = contextLength, stride = 4, batchSize = 8)

    val vocabSize = 50257 // R50K_BASE Größe
    val inputEmbedding =
        InputEmbeddingMultik(
            tokenEmbedding = TokenEmbeddingMultik(vocabSize, embeddingDim, seed = 42),
            positionalEmbedding = PositionalEmbeddingMultik(contextLength, embeddingDim, seed = 42),
        )

    val firstSample = loader.samples().first()
    val embeddings = inputEmbedding.forward(firstSample.inputIds)
    val attention = SelfAttentionMultik(embeddingDim = embeddingDim, dK = 64, seed = 42, causal = true)
    val context = attention.forward(embeddings) // [ctx, 64]

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [${embeddings.shape[0]}, ${embeddings.shape[1]}]")
    println("erstes Input-Embedding (5 Werte): ${embeddings[0].toList().take(5)}")
    println("attention output shape: [${context.shape[0]}, ${context.shape[1]}]")
}
