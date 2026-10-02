package ch.zuegi.ml.llm.kapitel5.training

import ch.zuegi.ml.llm.kapitel5.library.autograd.AdamOptimizer
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import kotlin.math.exp

class EarlyStoppingTrainerTest {
    @TempDir
    lateinit var tempDir: Path

    @Test
    fun `train stellt die Gewichte der besten Validierungsepoche wieder her`() {
        val model = tinyModel()
        val trainer = GPTTrainer(model, AdamOptimizer(model.parameters(), learningRate = LEARNING_RATE))
        val samples = repeatingSamples()
        val metrics = mutableListOf<EpochMetrics>()

        val result =
            EarlyStoppingTrainer(
                model = model,
                trainer = trainer,
                checkpointPath = tempDir.resolve("best.bin"),
                config = EarlyStoppingConfig(maxEpochs = MAX_EPOCHS, patience = PATIENCE),
            ).train(samples.drop(VALIDATION_COUNT), samples.take(VALIDATION_COUNT), BATCH_SIZE) { metrics += it }

        val best = metrics.single { it.epoch == result.bestEpoch }
        assertTrue(best.isBestEpoch)
        assertEquals(metrics.minOf { it.validationLoss }, result.bestValidationLoss)
        assertEquals(exp(best.validationLoss), best.validationPerplexity, TOLERANCE)
        assertEquals(result.bestValidationLoss, trainer.validate(samples.take(VALIDATION_COUNT), BATCH_SIZE), TOLERANCE)
    }

    companion object {
        private const val LEARNING_RATE = 0.05
        private const val MAX_EPOCHS = 6
        private const val PATIENCE = 2
        private const val VALIDATION_COUNT = 4
        private const val BATCH_SIZE = 4
        private const val TOLERANCE = 1e-12
    }
}
