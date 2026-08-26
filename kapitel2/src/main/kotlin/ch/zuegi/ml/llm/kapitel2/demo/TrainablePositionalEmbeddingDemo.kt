package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.library.TrainablePositionalEmbedding
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun main() {
    val positionalEmbedding = TrainablePositionalEmbedding(contextLength = 8, embeddingDim = 4, seed = 42)
    val sequenceLength = 4

    val position0Before = positionalEmbedding.weights[0].toList()
    println("Position 0 vor: $position0Before")

    positionalEmbedding.forward(sequenceLength)

    val gradOutput = mk.ones<Double>(sequenceLength, 4)
    positionalEmbedding.backward(gradOutput)
    positionalEmbedding.step(learningRate = 0.01)

    val position0After = positionalEmbedding.weights[0].toList()
    val diff = (position0Before zip position0After).map { (before, after) -> after - before }

    println("Mit echten Gradienten (ones):")
    println("Änderung pro Dimension: ${diff.take(4)}")
    println("Durchschnitt: ${diff.average()}")

    check(diff.all { it < 0.0 }) { "Erwartet negatives Update für Position 0, war aber: $diff" }
}

