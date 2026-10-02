package ch.zuegi.ml.llm.kapitel5.training

import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class GPTTrainerTest {
    @Test
    fun `mehrere Epochen senken den Trainings-Loss auf wiederholtem Text`() {
        val model = tinyModel()
        val trainer = GPTTrainer(model, AdamOptimizer(model.parameters(), learningRate = LEARNING_RATE))
        val samples = repeatingSamples()

        val initialLoss = trainer.validate(samples, BATCH_SIZE)
        repeat(EPOCHS) { trainer.trainEpoch(samples, BATCH_SIZE) }
        val finalLoss = trainer.validate(samples, BATCH_SIZE)

        assertTrue(finalLoss < initialLoss * LOSS_REDUCTION_FACTOR) {
            "Loss sank nicht deutlich: initial=$initialLoss final=$finalLoss"
        }
    }

    @Test
    fun `validate veraendert keine Gewichte`() {
        val model = tinyModel()
        val trainer = GPTTrainer(model, AdamOptimizer(model.parameters()))
        val before = model.parameterSnapshot()

        trainer.validate(repeatingSamples(), BATCH_SIZE)

        model.parameterSnapshot().zip(before).forEach { (after, original) ->
            assertArrayEquals(original, after)
        }
    }

    @Test
    fun `Gradienten eingefrorener Parameter beeinflussen das Clipping nicht`() {
        val reference = trainOnlyOutputProjection(frozenGradient = 0.0)
        val withLargeFrozenGradient = trainOnlyOutputProjection(frozenGradient = HUGE_GRADIENT)

        assertArrayEquals(reference, withLargeFrozenGradient)
    }

    private fun trainOnlyOutputProjection(frozenGradient: Double): DoubleArray {
        val model = tinyModel()
        model.tokenEmbedding.grad[0] = frozenGradient
        val trainer = GPTTrainer(model, AdamOptimizer(listOf(model.wOutput), learningRate = LEARNING_RATE))
        trainer.trainEpoch(repeatingSamples(), BATCH_SIZE)
        return DoubleArray(model.wOutput.size) { model.wOutput.data[it] }
    }

    companion object {
        private const val HUGE_GRADIENT = 1e12
        private const val LEARNING_RATE = 0.01
        private const val BATCH_SIZE = 4
        private const val EPOCHS = 15
        private const val LOSS_REDUCTION_FACTOR = 0.5
    }
}
