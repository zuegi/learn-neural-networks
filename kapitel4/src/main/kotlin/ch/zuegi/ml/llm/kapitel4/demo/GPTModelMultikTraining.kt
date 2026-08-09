package ch.zuegi.ml.llm.kapitel4.demo

import ch.zuegi.ml.llm.kapitel4.GPTConfig
import ch.zuegi.ml.llm.kapitel4.GPTModelMultikTensor
import ch.zuegi.ml.llm.kapitel4.library.autograd.SGDTensorMultik
import ch.zuegi.ml.llm.kapitel4.library.tokenize.GPT2Tokenizer
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText
import org.jetbrains.kotlinx.multik.ndarray.data.get

fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    val config =
        GPTConfig(
            vocabSize = tokenizer.vocabSize,
            contextLength = 4,
            embeddingDim = 64,
            numLayers = 2,
            numHeads = 8,
            seed = 42L,
        )

    val model = GPTModelMultikTensor(config)
    val loader =
        TextDataLoader(
            tokenIds = tokenIds,
            contextLength = config.contextLength,
            stride = config.contextLength,
        )

    val samples = loader.samples().take(20)
    val sgd = SGDTensorMultik(model.parameters(), learningRate = 0.05)

    val epochs = 10
    for (epoch in 1..epochs) {
        var epochLoss = 0.0
        for (sample in samples) {
            sgd.zeroGrad()
            val loss = model.loss(sample.inputIds, sample.targetIds)
            loss.backward()
            sgd.step()
            epochLoss += loss.data[0]
        }
        val avg = epochLoss / samples.size
        println("epoch $epoch/$epochs  loss=${"%.4f".format(avg)}")
    }

    val startIds = tokenIds.take(config.contextLength)
    val generated =
        model.generate(
            startIds,
            maxNewTokens = 20,
            temperature = 0.8,
            topK = 5,
            generatorSeed = 123,
        )

    println("start:     ${tokenizer.decode(startIds)}")
    println("generated: ${tokenizer.decode(generated)}")
}
