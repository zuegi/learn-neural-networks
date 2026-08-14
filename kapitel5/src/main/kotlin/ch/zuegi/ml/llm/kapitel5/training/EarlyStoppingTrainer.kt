package ch.zuegi.ml.llm.kapitel5.training

import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.shared.TrainingSample
import java.nio.file.Path
import kotlin.time.measureTime

data class EarlyStoppingConfig(
    val maxEpochs: Int,
    val patience: Int,
    val minDelta: Double = 0.0,
)

data class EpochMetrics(
    val epoch: Int,
    val trainLoss: Double,
    val validationLoss: Double,
    val trainPerplexity: Double,
    val validationPerplexity: Double,
    val epochSeconds: Long,
    val isBestEpoch: Boolean,
)

data class EarlyStoppingResult(
    val bestEpoch: Int,
    val bestValidationLoss: Double,
    val stoppedEarly: Boolean,
)

class EarlyStoppingTrainer(
    private val model: GPTModelMultikTensor,
    private val trainer: GPTTrainer,
    private val checkpointPath: Path,
    private val config: EarlyStoppingConfig,
) {
    fun train(
        trainingSamples: List<TrainingSample>,
        validationSamples: List<TrainingSample>,
        batchSize: Int,
        onEpochFinished: (EpochMetrics) -> Unit = {},
    ): EarlyStoppingResult {
        require(config.maxEpochs > 0) { "maxEpochs muss > 0 sein" }
        require(config.patience > 0) { "patience muss > 0 sein" }
        require(config.minDelta >= 0.0) { "minDelta muss >= 0.0 sein" }

        var bestValidationLoss = Double.POSITIVE_INFINITY
        var bestEpoch = 0
        var epochsWithoutImprovement = 0
        var stoppedEarly = false

        for (epoch in 1..config.maxEpochs) {
            var trainLoss = 0.0
            var validationLoss = 0.0

            val epochDuration =
                measureTime {
                    trainLoss = trainer.trainEpoch(trainingSamples, batchSize)
                    validationLoss = trainer.validate(validationSamples, batchSize)
                }

            val isBestEpoch = validationLoss < bestValidationLoss - config.minDelta
            if (isBestEpoch) {
                bestValidationLoss = validationLoss
                bestEpoch = epoch
                epochsWithoutImprovement = 0
                saveModelWeights(model, checkpointPath)
            } else {
                epochsWithoutImprovement += 1
            }

            onEpochFinished(
                EpochMetrics(
                    epoch = epoch,
                    trainLoss = trainLoss,
                    validationLoss = validationLoss,
                    trainPerplexity = kotlin.math.exp(trainLoss),
                    validationPerplexity = kotlin.math.exp(validationLoss),
                    epochSeconds = epochDuration.inWholeSeconds,
                    isBestEpoch = isBestEpoch,
                ),
            )

            if (epochsWithoutImprovement >= config.patience) {
                stoppedEarly = true
                break
            }
        }

        require(bestEpoch > 0) { "Kein gueltiger Checkpoint waehrend Training erstellt" }
        loadModelWeights(model, checkpointPath)

        return EarlyStoppingResult(
            bestEpoch = bestEpoch,
            bestValidationLoss = bestValidationLoss,
            stoppedEarly = stoppedEarly,
        )
    }
}

