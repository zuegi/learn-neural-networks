package ch.zuegi.ml.llm.kapitel5.demo

import ch.zuegi.ml.llm.kapitel4.library.autograd.TensorMultik
import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import ch.zuegi.ml.llm.kapitel5.model.GPTConfig
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel5.training.GPTTrainer
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.TrainingSample
import ch.zuegi.ml.llm.shared.readVerdictText
import ch.zuegi.ml.llm.shared.tokenize.GPT2Tokenizer
import kotlin.math.exp
import kotlin.time.measureTime

private data class Variant(
    val name: String,
    val trainableParameters: (GPTModelMultikTensor) -> List<TensorMultik>,
)

private data class EpochLoss(
    val train: Double,
    val validation: Double,
)

private data class VariantResult(
    val name: String,
    val trainableCount: Int,
    val initialValidationLoss: Double,
    val epochs: List<EpochLoss>,
    val seconds: Long,
)

private val variants =
    listOf(
        Variant("nur W_out") { listOf(it.wOutput) },
        Variant("Embeddings statisch") { model ->
            model.parameters().filterNot { it === model.tokenEmbedding || it === model.positionalEmbedding }
        },
        Variant("alle Parameter") { it.parameters() },
    )

/**
 * Vergleicht, wie weit das Modell kommt, wenn nur ein Teil der Gewichte lernt.
 * Alle Varianten starten mit identischen Gewichten (gleicher Seed) und sehen
 * dieselben Daten; sie unterscheiden sich nur in der Parameterliste des Optimizers.
 */
fun main() {
    val text = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val samples = TextDataLoader(tokenizer.encode(text), CONTEXT_LENGTH, stride = CONTEXT_LENGTH).samples()
    val (training, validation) = splitSamples(samples)
    val config = comparisonConfig(tokenizer.vocabSize)
    println("vocab=${tokenizer.vocabSize} train=${training.size} val=${validation.size} epochs=$EPOCHS")

    val results = variants.map { runVariant(it, config, training, validation) }
    printSummary(results)
}

private fun comparisonConfig(vocabSize: Int) =
    GPTConfig(
        vocabSize = vocabSize,
        contextLength = CONTEXT_LENGTH,
        embeddingDim = EMBEDDING_DIM,
        numLayers = NUM_LAYERS,
        numHeads = NUM_HEADS,
        dropoutProb = DROPOUT,
        seed = SEED,
    )

private fun runVariant(
    variant: Variant,
    config: GPTConfig,
    training: List<TrainingSample>,
    validation: List<TrainingSample>,
): VariantResult {
    val model = GPTModelMultikTensor(config)
    val parameters = variant.trainableParameters(model)
    val trainer = GPTTrainer(model, AdamOptimizer(parameters, learningRate = LEARNING_RATE))
    val initialValidationLoss = trainer.validate(validation, BATCH_SIZE)
    val epochs = mutableListOf<EpochLoss>()
    val duration =
        measureTime {
            repeat(EPOCHS) { epoch ->
                epochs += EpochLoss(trainer.trainEpoch(training, BATCH_SIZE), trainer.validate(validation, BATCH_SIZE))
                println("${variant.name} epoch ${epoch + 1}: ${format(epochs.last())}")
            }
        }
    val trainableCount = parameters.sumOf { it.size }
    return VariantResult(variant.name, trainableCount, initialValidationLoss, epochs, duration.inWholeSeconds)
}

private fun splitSamples(samples: List<TrainingSample>): Pair<List<TrainingSample>, List<TrainingSample>> {
    val validationCount = (samples.size * VALIDATION_RATIO).toInt()
    return samples.dropLast(validationCount) to samples.takeLast(validationCount)
}

private fun printSummary(results: List<VariantResult>) {
    println()
    println("| Variante | trainierbare Parameter | Val-Loss Start | bester Val-Loss | beste Val-PPL | Train-Loss Ende | Sekunden |")
    println("|---|---|---|---|---|---|---|")
    results.forEach { result ->
        val bestValidation = result.epochs.minOf { it.validation }
        println(
            "| ${result.name} | ${result.trainableCount} | ${"%.3f".format(result.initialValidationLoss)} " +
                "| ${"%.3f".format(bestValidation)} | ${"%.1f".format(exp(bestValidation))} " +
                "| ${"%.3f".format(result.epochs.last().train)} | ${result.seconds} |",
        )
    }
}

private fun format(loss: EpochLoss) = "train=${"%.4f".format(loss.train)} val=${"%.4f".format(loss.validation)}"

private const val CONTEXT_LENGTH = 32
private const val EMBEDDING_DIM = 32
private const val NUM_LAYERS = 1
private const val NUM_HEADS = 2
private const val DROPOUT = 0.1
private const val SEED = 42L
private const val LEARNING_RATE = 0.003
private const val BATCH_SIZE = 4
private const val EPOCHS = 6
private const val VALIDATION_RATIO = 0.2
