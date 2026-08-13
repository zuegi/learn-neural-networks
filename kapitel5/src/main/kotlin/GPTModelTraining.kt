import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import ch.zuegi.ml.llm.kapitel5.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.kapitel5.model.GPTConfig
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel5.model.GenerationConfig
import ch.zuegi.ml.llm.kapitel5.training.GPTTrainer
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.TrainingSample
import ch.zuegi.ml.llm.shared.readVerdictText
import java.time.LocalTime
import kotlin.time.measureTime

fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val trainingSampleSize = 500
    val learningRate = 0.001
    val epochs = 10
    val batchSize = 2

    val generationConfig =
        GenerationConfig(
            maxNewTokens = 4,
            temperature = 0.9,
            topK = 20,
            generatorSeed = 123,
            greedy = false,
        )

    val config =
        GPTConfig(
            vocabSize = tokenizer.vocabSize,
            contextLength = 32,
            embeddingDim = 32,
            numLayers = 1,
            numHeads = 2,
            dropoutProb = 0.1,
            useQkvBias = false,
            useOutputBias = false,
            seed = 42L,
        )

    val model = GPTModelMultikTensor(config)
    val loader =
        TextDataLoader(
            tokenIds = tokenIds,
            contextLength = config.contextLength,
            stride = 1,
        )

    val samples = loader.samples().take(trainingSampleSize)
    val (trainingSamples, validationSamples) = splitSamples(samples)
    val optimizer = AdamOptimizer(model.parameters(), learningRate = learningRate)
    val trainer = GPTTrainer(model, optimizer)

    println("${LocalTime.now()} - Start calculation $epochs epochs")
    val timeDuration =
        measureTime {
            for (epoch in 1..epochs) {
                var trainLoss = 0.0
                var validationLoss = 0.0

                val epochMeasureTime =
                    measureTime {
                        trainLoss = trainer.trainEpoch(trainingSamples, batchSize)
                        validationLoss = trainer.validate(validationSamples, batchSize)
                    }

                val trainPpl = kotlin.math.exp(trainLoss)
                val validationPpl = kotlin.math.exp(validationLoss)

                println(
                    "${LocalTime.now()} - epoch $epoch/$epochs " +
                        "train=${"%.4f".format(trainLoss)} " +
                        "val=${"%.4f".format(validationLoss)} " +
                        "train_ppl=${"%.2f".format(trainPpl)} " +
                        "val_ppl=${"%.2f".format(validationPpl)} " +
                        "epoch_time=${epochMeasureTime.inWholeSeconds}s",
                )
            }
        }

    val startIds = tokenIds.take(config.contextLength)
    val generated =
        model.generate(
            startIds,
            generationConfig,
        )

    println("${LocalTime.now()} - Zeit des Trainings: ${timeDuration.inWholeSeconds} Sekunden")
    println("${LocalTime.now()} - start:     ${tokenizer.decode(startIds)}")
    println("${LocalTime.now()} - generated: ${tokenizer.decode(generated)}")
}

private fun splitSamples(samples: List<TrainingSample>): Pair<List<TrainingSample>, List<TrainingSample>> {
    require(samples.size >= 2) { "Mindestens 2 Samples für Training und Validation nötig" }

    val validationCount = (samples.size * VALIDATION_RATIO).toInt().coerceIn(1, samples.lastIndex)
    val trainingSamples = samples.dropLast(validationCount)
    val validationSamples = samples.takeLast(validationCount)
    return trainingSamples to validationSamples
}

private const val VALIDATION_RATIO = 0.2
