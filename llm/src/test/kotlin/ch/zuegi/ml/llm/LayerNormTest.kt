package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.sqrt

class LayerNormTest {
    @Test
    fun `output rows have mean approx zero and variance approx one`() {
        val embeddingDim = 4
        val layerNorm = LayerNorm(embeddingDim)
        val input =
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0, 4.0),
                doubleArrayOf(10.0, 20.0, 30.0, 40.0),
            )

        val output = layerNorm.forward(input)

        for (row in output) {
            val mean = row.average()
            val variance = row.sumOf { (it - mean) * (it - mean) } / row.size

            assertThat(mean).isCloseTo(0.0, offset(1e-9))
            assertThat(variance).isCloseTo(1.0, offset(1e-4))
        }
    }

    @Test
    fun `output has same shape as input`() {
        val layerNorm = LayerNorm(embeddingDim = 3)
        val input =
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
            )

        val output = layerNorm.forward(input)

        assertThat(output.size).isEqualTo(2)
        assertThat(output.all { it.size == 3 }).isTrue()
    }

    @Test
    fun `gamma and beta transform the normalized output`() {
        val embeddingDim = 4
        val layerNorm = LayerNorm(embeddingDim)

        // gamma=2, beta=1 -> jede normalisierte Zeile wird skaliert und verschoben
        for (i in 0 until embeddingDim) {
            layerNorm.gamma[i] = 2.0
            layerNorm.beta[i] = 1.0
        }

        val input = arrayOf(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val output = layerNorm.forward(input)

        // Nach Transformation: mean = beta = 1, std = gamma = 2
        val mean = output[0].average()
        val variance = output[0].sumOf { (it - mean) * (it - mean) } / output[0].size

        assertThat(mean).isCloseTo(1.0, offset(1e-9))
        assertThat(sqrt(variance)).isCloseTo(2.0, offset(1e-4))
    }
}
