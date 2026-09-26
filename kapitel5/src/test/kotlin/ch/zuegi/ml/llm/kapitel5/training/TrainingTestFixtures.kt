package ch.zuegi.ml.llm.kapitel5.training

import ch.zuegi.ml.llm.kapitel5.model.GPTConfig
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.shared.TextDataLoader
import ch.zuegi.ml.llm.shared.TrainingSample
import org.jetbrains.kotlinx.multik.ndarray.data.get

internal const val TEST_CONTEXT_LENGTH = 4

internal fun tinyModel(seed: Long = 3L): GPTModelMultikTensor =
    GPTModelMultikTensor(
        GPTConfig(
            vocabSize = 10,
            contextLength = TEST_CONTEXT_LENGTH,
            embeddingDim = 8,
            numLayers = 1,
            numHeads = 2,
            dropoutProb = 0.0,
            seed = seed,
        ),
    )

internal fun repeatingSamples(): List<TrainingSample> =
    TextDataLoader(
        tokenIds = List(24) { it % 5 },
        contextLength = TEST_CONTEXT_LENGTH,
    ).samples()

internal fun GPTModelMultikTensor.parameterSnapshot(): List<DoubleArray> =
    parameters().map { p -> DoubleArray(p.size) { p.data[it] } }

