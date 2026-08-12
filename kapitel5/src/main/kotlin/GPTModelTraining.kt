import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import ch.zuegi.ml.llm.kapitel5.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.kapitel5.model.GPTConfig
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel5.model.GenerationConfig
import ch.zuegi.ml.llm.kapitel5.training.GPTTrainer
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import java.time.LocalTime
import kotlin.time.measureTime

fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val trainingSampleSize = 50
    val learningRate = 0.01
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
            embeddingDim = 64,
            numLayers = 2,
            numHeads = 4,
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
            stride = config.contextLength,
        )

    val samples = loader.samples().take(trainingSampleSize)
    val optimizer = AdamOptimizer(model.parameters(), learningRate = learningRate)
    val trainer = GPTTrainer(model, optimizer)

    println("${LocalTime.now()} - Start calculation epochs")
    val timeDuration =
        measureTime {
            for (epoch in 1..epochs) {
                val avgLoss = trainer.trainEpoch(samples, batchSize)
                println("${LocalTime.now()} - epoch $epoch/$epochs  loss=${"%.4f".format(avgLoss)}")
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
