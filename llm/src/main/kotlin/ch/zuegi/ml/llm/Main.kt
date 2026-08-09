package ch.zuegi.ml.llm

import ch.zuegi.ml.llm.autograd.GPTModelLayer
import ch.zuegi.ml.llm.autograd.SGD
import java.nio.charset.StandardCharsets

fun main() {
//    mainSimple()
    mainTrain()
}

fun mainSimple() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 256

    val model =
        GPTModel(
            vocabSize = tokenizer.vocabSize,
            contextLength = contextLength,
            embeddingDim = embeddingDim,
            numLayers = 2,
            numHeads = 8,
            causal = true,
            seed = 42,
        )

    // Start-Sequenz: erste contextLength Tokens des Textes
    val startIds = tokenIds.take(contextLength)

    // Forward-Pass: Logits der letzten Position
    val logits = model.forward(startIds)
    println("start: ${tokenizer.decode(startIds)}")
    println("logits shape: [${logits.size}, ${logits[0].size}]")

    // Autoregressive Generierung (Modell ist untrainiert -> Ausgabe ist Kauderwelsch)
    val generated = model.generate(startIds, maxNewTokens = 20, greedy = true)
    println("generated ids: $generated")
    println("generated text: ${tokenizer.decode(generated)}")
}

fun mainTrain() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    val contextLength = 4
    val embeddingDim = 32
    val numHeads = 2

    val model =
        GPTModelLayer(
            vocabSize = tokenizer.vocabSize,
            contextLength = contextLength,
            embeddingDim = embeddingDim,
            numLayers = 2,
            numHeads = numHeads,
            dK = embeddingDim / numHeads,
            causal = true,
            seed = 42,
        )

    val loader =
        TextDataLoader(
            tokenIds = tokenIds,
            contextLength = contextLength,
            stride = contextLength,
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

    val startIds = tokenIds.take(contextLength)
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

fun readVerdictText(): String {
    val resourcePath = "/text/the-verdict.txt"
    val stream =
        object {}.javaClass.getResourceAsStream(resourcePath)
            ?: error("Ressource nicht gefunden: $resourcePath")

    return stream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
}
