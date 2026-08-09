import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import ch.zuegi.ml.llm.kapitel5.library.autograd.TensorMultik
import ch.zuegi.ml.llm.kapitel5.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.kapitel5.model.GPTConfig
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel5.model.GenerationConfig
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
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
            maxNewTokens = 10,
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
    println("${LocalTime.now()} - Start calculation epochs")
    val timeDuration =
        measureTime {
            for (epoch in 1..epochs) {
                var epochLoss = 0.0
                var batchCount = 0
                var index = 0

                while (index < samples.size) {
                    val end = minOf(index + batchSize, samples.size)
                    val batch = samples.subList(index, end)

                    optimizer.zeroGrad()

                    var batchLoss = 0.0
                    for (sample in batch) {
                        val loss = model.loss(sample.inputIds, sample.targetIds)
                        loss.backward()
                        batchLoss += loss.data[0]
                    }

                    clipGradients(model.parameters(), maxNorm = 1.0)
                    optimizer.step()

                    epochLoss += batchLoss
                    batchCount += 1
                    index = end
                }

                val avgLoss = epochLoss / batchCount
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

fun clipGradients(
    parameters: List<TensorMultik>,
    maxNorm: Double = 1.0,
) {
    var norm = 0.0
    for (param in parameters) {
        for (i in 0 until param.size) {
            norm += param.grad[i] * param.grad[i]
        }
    }
    norm = kotlin.math.sqrt(norm)

    if (norm > maxNorm) {
        val scale = maxNorm / norm
        for (param in parameters) {
            for (i in 0 until param.size) {
                param.grad[i] *= scale
            }
        }
    }
}
