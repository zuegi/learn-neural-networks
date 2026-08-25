package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.library.TrainableTokenEmbedding
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val embedding = TrainableTokenEmbedding(vocabSize = 100, embeddingDim = 8, seed = 42)
    val tokenIds = listOf(3, 7, 3, 12)

    // zufällig initalisierte Gewichte vor dem Training
    val weightsBefore = embedding.weights[3].toList()
    println("Token 3 vor: $weightsBefore")

    // Forward: merkt sich die Token-IDs für den nachfolgenden Backward-Pass
    embedding.forward(tokenIds)

    // Echte Gradienten (statt zeros): alle 1.0
    val gradOutput = mk.ones<Double>(4, 8)
    embedding.backward(gradOutput)
    embedding.step(learningRate = 0.01)

    // Gewichte nach training
    val weightsAfter = embedding.weights[3].toList()
    val diff = (weightsBefore zip weightsAfter).map { (before, after) -> after - before }

    println("Mit echten Gradienten (ones):")
    println("Änderung pro Dimension: ${diff.take(4)}") // erste 3 Werte
    println("Durchschnitt: ${diff.average()}") // sollte etwa -0.02 sein (Token 3 kommt zweimal vor)
}
