package ch.zuegi.ml.llm.kapitel4.demo

import ch.zuegi.ml.llm.kapitel4.GPTConfig
import ch.zuegi.ml.llm.kapitel4.GPTModel
import ch.zuegi.ml.llm.kapitel4.scratch.autograd.SGD
import ch.zuegi.ml.llm.shared.tokenize.SimpleTokenizerV1
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.readVerdictText

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
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
    val model = GPTModel(config)
    val loader =
        TextDataLoader(
            tokenIds = tokenIds,
            contextLength = config.contextLength,
            stride = config.contextLength,
        )

    // Wenige Samples, damit das Scalar-Autograd auf der CPU in vertretbarer Zeit trainiert
    val samples = loader.samples().take(20)
    val sgd = SGD(model.parameters(), learningRate = 0.05)

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
            maxNewTokens = 10,
            temperature = 0.8,
            topK = 5,
            generatorSeed = 123,
        )
    println("start:     ${tokenizer.decode(startIds)}")
    println("generated: ${tokenizer.decode(generated)}")
}
