package ch.zuegi.ml.llm.kapitel5.training

import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import ch.zuegi.ml.llm.kapitel5.library.autograd.TensorMultik
import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import ch.zuegi.ml.llm.shared.TrainingSample
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set

class GPTTrainer(
    private val model: GPTModelMultikTensor,
    private val optimizer: AdamOptimizer,
    private val maxGradNorm: Double = 1.0,
) {
    fun trainEpoch(
        samples: List<TrainingSample>,
        batchSize: Int,
    ): Double {
        var totalSampleLoss = 0.0
        var sampleCount = 0
        var index = 0

        while (index < samples.size) {
            val end = minOf(index + batchSize, samples.size)
            val batch = samples.subList(index, end)

            optimizer.zeroGrad()

            for (sample in batch) {
                val loss =
                    model.loss(
                        tokenIds = sample.inputIds,
                        targetIds = sample.targetIds,
                        training = true,
                    )
                loss.backward()
                totalSampleLoss += loss.data[0]
                sampleCount += 1
            }

            clipGradients(model.parameters())
            optimizer.step()
            index = end
        }

        return totalSampleLoss / sampleCount
    }

    fun validate(
        samples: List<TrainingSample>,
        batchSize: Int,
    ): Double {
        var totalSampleLoss = 0.0
        var sampleCount = 0
        var index = 0

        while (index < samples.size) {
            val end = minOf(index + batchSize, samples.size)
            val batch = samples.subList(index, end)

            for (sample in batch) {
                val loss =
                    model.loss(
                        tokenIds = sample.inputIds,
                        targetIds = sample.targetIds,
                        training = false,
                    )
                totalSampleLoss += loss.data[0]
                sampleCount += 1
            }

            index = end
        }

        return totalSampleLoss / sampleCount
    }

    private fun clipGradients(parameters: List<TensorMultik>) {
        var norm = 0.0
        for (param in parameters) {
            for (i in 0 until param.size) {
                norm += param.grad[i] * param.grad[i]
            }
        }
        norm = kotlin.math.sqrt(norm)

        if (norm > maxGradNorm) {
            val scale = maxGradNorm / norm
            for (param in parameters) {
                for (i in 0 until param.size) {
                    param.grad[i] *= scale
                }
            }
        }
    }
}
