import ch.zuegi.ml.llm.kapitel5.library.autograd.SGDTensorMultik
import ch.zuegi.ml.llm.kapitel5.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.kapitel5.model.GPTConfig
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel5.model.GenerationConfig
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.ndarray.data.get

fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val trainingSampleSize = 100
    val learningRate = 0.05
    val epochs = 10
    val batchSize = 1

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
    val sgd = SGDTensorMultik(model.parameters(), learningRate = learningRate)

    for (epoch in 1..epochs) {
        var epochLoss = 0.0
        var batchCount = 0

        val sampleList = samples.toList()
        var index = 0

        while (index < sampleList.size) {
            val end = minOf(index + batchSize, sampleList.size)
            val batch = sampleList.subList(index, end)

            sgd.zeroGrad()

            var batchLossSum = 0.0
            for (sample in batch) {
                val loss = model.loss(sample.inputIds, sample.targetIds) // TensorMultik (Skalar)
                loss.backward() // Gradienten akkumulieren
                batchLossSum += loss.data[0]
            }

            // Mittelwert innerhalb Batch für Logging
            val batchAvgLoss = batchLossSum / batch.size
            epochLoss += batchAvgLoss
            batchCount += 1

            // EIN Optimizer-Step pro Batch
            sgd.step()

            index = end
        }

        val avg = epochLoss / batchCount
        println("epoch $epoch/$epochs  loss=${"%.4f".format(avg)}")
    }

    val startIds = tokenIds.take(config.contextLength)
    val generated =
        model.generate(
            startIds,
            generationConfig,
        )

    println("start:     ${tokenizer.decode(startIds)}")
    println("generated: ${tokenizer.decode(generated)}")
}
