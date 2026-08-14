import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import ch.zuegi.ml.llm.kapitel5.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.kapitel5.model.GPTConfig
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel5.model.GenerationConfig
import ch.zuegi.ml.llm.kapitel5.training.EarlyStoppingConfig
import ch.zuegi.ml.llm.kapitel5.training.EarlyStoppingResult
import ch.zuegi.ml.llm.kapitel5.training.EarlyStoppingTrainer
import ch.zuegi.ml.llm.kapitel5.training.GPTTrainer
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.TrainingSample
import ch.zuegi.ml.llm.shared.readTheVerdictText
import java.nio.file.Path
import java.time.LocalTime
import kotlin.time.measureTime

fun main() {
    val verdictText = readTheVerdictText()
    println("Anzahl Zeichen von rawText: ${verdictText.corpusSizeInCharacters()}")
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(verdictText.text)

    val trainingSampleSize = 500
    val learningRate = 0.001
    val epochs = 10
    val batchSize = 2
    val patience = 3

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
    val earlyStoppingTrainer =
        EarlyStoppingTrainer(
            model = model,
            trainer = trainer,
            checkpointPath = Path.of("kapitel5/target/checkpoints/best-model.bin"),
            config = EarlyStoppingConfig(maxEpochs = epochs, patience = patience, minDelta = MIN_DELTA),
        )

    println("${LocalTime.now()} - Start calculation $epochs epochs")
    var trainingResult: EarlyStoppingResult? = null
    val timeDuration =
        measureTime {
            trainingResult =
                earlyStoppingTrainer.train(
                    trainingSamples = trainingSamples,
                    validationSamples = validationSamples,
                    batchSize = batchSize,
                ) { metrics ->
                    println(
                        "${LocalTime.now()} - epoch ${metrics.epoch}/$epochs " +
                            "train=${"%.4f".format(metrics.trainLoss)} " +
                            "val=${"%.4f".format(metrics.validationLoss)} " +
                            "train_ppl=${"%.2f".format(metrics.trainPerplexity)} " +
                            "val_ppl=${"%.2f".format(metrics.validationPerplexity)} " +
                            "epoch_time=${metrics.epochSeconds}s",
                    )
                }
        }
    val result = requireNotNull(trainingResult)

    val startIds = tokenIds.take(config.contextLength)
    val generated =
        model.generate(
            startIds,
            generationConfig,
        )

    println("${LocalTime.now()} - Zeit des Trainings: ${timeDuration.inWholeSeconds} Sekunden")
    println(
        "${LocalTime.now()} - best_epoch=${result.bestEpoch} best_val=${"%.4f".format(result.bestValidationLoss)} " +
            "stopped_early=${result.stoppedEarly}",
    )
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
private const val MIN_DELTA = 0.0001
