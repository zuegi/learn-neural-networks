package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.library.TrainableInputEmbedding
import ch.zuegi.ml.llm.kapitel2.library.TrainablePositionalEmbedding
import ch.zuegi.ml.llm.kapitel2.library.TrainableTokenEmbedding
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val tokenEmbedding = TrainableTokenEmbedding(vocabSize = 100, embeddingDim = 4, seed = 42)
    val positionalEmbedding = TrainablePositionalEmbedding(contextLength = 8, embeddingDim = 4, seed = 42)
    val inputEmbedding = TrainableInputEmbedding(tokenEmbedding, positionalEmbedding)
    val tokenIds = listOf(3, 7, 3, 12)

    val token3Before = tokenEmbedding.weights[3].toList()
    val position0Before = positionalEmbedding.weights[0].toList()

    println("Token 3 vor: $token3Before")
    println("Position 0 vor: $position0Before")

    val inputVectors = inputEmbedding.forward(tokenIds)
    println("input shape: [${inputVectors.shape[0]}, ${inputVectors.shape[1]}]")

    val gradOutput = mk.ones<Double>(tokenIds.size, 4)
    inputEmbedding.backward(gradOutput)
    inputEmbedding.step(learningRate = 0.01)

    val token3After = tokenEmbedding.weights[3].toList()
    val position0After = positionalEmbedding.weights[0].toList()

    val token3Diff = (token3Before zip token3After).map { (before, after) -> after - before }
    val position0Diff = (position0Before zip position0After).map { (before, after) -> after - before }

    println("Mit echten Gradienten (ones):")
    println("Token 3 Änderung: ${token3Diff.take(4)}")
    println("Position 0 Änderung: ${position0Diff.take(4)}")

    check(token3Diff.all { it < 0.0 }) { "Erwartet negatives Update für Token 3, war aber: $token3Diff" }
    check(position0Diff.all { it < 0.0 }) {
        "Erwartet negatives Update für Position 0, war aber: $position0Diff"
    }
}

