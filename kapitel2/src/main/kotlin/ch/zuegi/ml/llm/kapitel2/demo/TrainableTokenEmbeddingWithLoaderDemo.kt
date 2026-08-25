package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.library.R50kBpeTokenizer
import ch.zuegi.ml.llm.kapitel2.library.TrainableTokenEmbedding
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.math.abs

/**
 * Demo: TrainableTokenEmbedding mit TextDataLoader trainieren.
 *
 * Zeigt bei jeder Iteration:
 * - welche Tokens verarbeitet werden
 * - wie sich deren Gewichte verändern
 * - Gesamt-Statistik am Ende
 */
fun main() {
    println("=== TrainableTokenEmbedding Training mit TextDataLoader ===\n")

    // End-to-End Setup: Text -> Tokenizer -> DataLoader -> trainierbares Embedding
    val rawText = readVerdictText()
    val tokenizer = R50kBpeTokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 8
    val learningRate = 0.01
    val maxIterations = 8 // für eine lesbare Konsole bewusst begrenzt

    val loader =
        TextDataLoader(
            tokenIds = tokenIds,
            contextLength = contextLength,
            stride = 1,
            batchSize = 1,
        )

    val embedding =
        TrainableTokenEmbedding(
            vocabSize = tokenizer.vocabSize,
            embeddingDim = embeddingDim,
            seed = 42,
        )

    val vocabSize = embedding.vocabSize

    // Speichere Gewichte vor dem Training
    val weightsBefore =
        Array(vocabSize) { v ->
            embedding.weights[v].toList().toDoubleArray()
        }

    // Snapshot für Delta-Messung zwischen zwei Iterationen
    var weightsPrevious =
        Array(vocabSize) { v ->
            embedding.weights[v].toList().toDoubleArray()
        }

    println("Textlänge: ${rawText.length}")
    println("Token-Anzahl: ${tokenIds.size}")
    println("Erste 12 Token-IDs: ${tokenIds.take(12)}")
    println("Vokabulargröße: $vocabSize, Embedding-Dimension: $embeddingDim")
    println("contextLength: $contextLength")
    println("Learning-Rate: $learningRate\n")

    // Training
    val samples = loader.samples().take(maxIterations)
    var iteration = 0
    val csvIterationRows = mutableListOf("iteration,tokenId,avgDeltaIteration,avgDeltaTotal")

    for (sample in samples) {
        iteration++
        println("--- Iteration $iteration ---")
        println("Input-IDs: ${sample.inputIds}")

        // Forward
        val output = embedding.forward(sample.inputIds)
        println("Output-Shape: [${output.shape[0]}, ${output.shape[1]}]")
        println("Target-IDs: ${sample.targetIds}")

        // Vereinfachte Loss: Gradient = 1.0 (simuliert, dass das Modell lernen kann)
        val gradOutput = mk.ndarray(List(output.shape[0]) { List(output.shape[1]) { 1.0 } })

        // Backward + Step
        embedding.backward(gradOutput)

        // Zeige Gradienten vor dem Update
        val affectedTokens = sample.inputIds.toSet()
        println("Betroffene Tokens: $affectedTokens")

        embedding.step(learningRate = learningRate)

        // Zeige Gewichtsveränderungen für betroffene Tokens
        // - Delta: Änderung seit letzter Iteration
        // - Kumuliert: Änderung seit Startzustand
        for (tokenId in affectedTokens) {
            val before = weightsBefore[tokenId]
            val previous = weightsPrevious[tokenId]
            val after = embedding.weights[tokenId].toList().toDoubleArray()

            val avgDeltaIteration = (previous zip after).map { abs(it.second - it.first) }.average()
            val avgDeltaTotal = (before zip after).map { abs(it.second - it.first) }.average()

            println(
                "  Token $tokenId: Ø-Δ Iteration = %.6f | Ø-Δ kumuliert = %.6f".format(
                    avgDeltaIteration,
                    avgDeltaTotal,
                ),
            )

            csvIterationRows +=
                "%d,%d,%.6f,%.6f".format(
                    iteration,
                    tokenId,
                    avgDeltaIteration,
                    avgDeltaTotal,
                )
        }

        // Snapshot für nächste Iteration aktualisieren
        weightsPrevious =
            Array(vocabSize) { v ->
                embedding.weights[v].toList().toDoubleArray()
            }
        println()
    }

    // Zusammenfassung
    println("\n=== Gesamtübersicht nach Training ===\n")
    println("Insgesamt $iteration Iterationen ausgeführt.\n")

    var totalAbsChange = 0.0
    var changedTokens = 0
    val csvSummaryRows = mutableListOf("tokenId,changed,sumAbsChange")

    for (v in 0 until vocabSize) {
        val before = weightsBefore[v]
        val after = embedding.weights[v].toList().toDoubleArray()
        val changes = (before zip after).map { it.second - it.first }
        val absChange = changes.map { abs(it) }.sum()

        if (absChange > 1e-6) {
            changedTokens++
            totalAbsChange += absChange
            println("Token $v:")
            println("  Gewicht-Änderung: ${changes.map { "%.6f".format(it) }}")
            println("  Summe |Änderungen|: %.6f".format(absChange))
            csvSummaryRows += "%d,true,%.6f".format(v, absChange)
        } else {
            csvSummaryRows += "%d,false,%.6f".format(v, absChange)
        }
    }

    println("\n--- Statistik ---")
    println("Tokens mit Änderungen: $changedTokens / $vocabSize")
    println(
        "Durchschnittliche Änderung pro Token: %.6f".format(
            if (changedTokens > 0) totalAbsChange / changedTokens else 0.0,
        ),
    )
    println("Gesamte Summe |Änderungen|: %.6f".format(totalAbsChange))
}
