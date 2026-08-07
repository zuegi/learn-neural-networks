package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.library.TokenEmbeddingMultik
import ch.zuegi.ml.llm.kapitel2.scratch.SimpleTokenizerV1
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val loader = TextDataLoader(tokenIds, contextLength = 4, stride = 4, batchSize = 8)
    val embedding = TokenEmbeddingMultik(vocabSize = tokenizer.vocabSize, embeddingDim = 256)

    println("shape/tabelle: ${embedding.weights.shape.toList()}")
    println("tokenId=2: ${embedding.lookup(2)}")
    println("tokenIds=[1,5,2]: ${embedding.lookup(listOf(1, 5, 2))}")
    println("***")

    val firstSample = loader.samples().first()
    val inputEmbeddings = embedding.lookup(firstSample.inputIds)

    val rows = inputEmbeddings.shape[0]
    val cols = inputEmbeddings.shape[1]
    val firstRowPreview = inputEmbeddings[0].toList().take(5)

    println("inputIds: ${firstSample.inputIds}")
    println("shape: [$rows, $cols]")
    println("erstes Token-Embedding (5 Werte): $firstRowPreview")
}
