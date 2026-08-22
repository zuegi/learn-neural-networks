package ch.zuegi.ml.llm.shared

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class TextDataLoaderTest {
    @Test
    fun `samples builds shifted input and target windows`() {
        val loader =
            TextDataLoader(
                tokenIds = listOf(10, 11, 12, 13, 14),
                contextLength = 2,
                stride = 1,
                batchSize = 1,
            )

        val samples = loader.samples()

        assertThat(samples).hasSize(3)
        assertThat(samples[0].inputIds).isEqualTo(listOf(10, 11))
        assertThat(samples[0].targetIds).isEqualTo(listOf(11, 12))

        assertThat(samples[1].inputIds).isEqualTo(listOf(11, 12))
        assertThat(samples[1].targetIds).isEqualTo(listOf(12, 13))

        assertThat(samples[2].inputIds).isEqualTo(listOf(12, 13))
        assertThat(samples[2].targetIds).isEqualTo(listOf(13, 14))
    }

    @Test
    fun `size returns number of windows with stride`() {
        val loader =
            TextDataLoader(
                tokenIds = listOf(0, 1, 2, 3, 4, 5, 6),
                contextLength = 3,
                stride = 2,
                batchSize = 1,
            )

        assertThat(loader.size()).isEqualTo(2)
        // starts: 0, 2
    }

    @Test
    fun `batches chunks samples by batchSize and keeps remainder`() {
        val loader =
            TextDataLoader(
                tokenIds = listOf(0, 1, 2, 3, 4, 5),
                contextLength = 2,
                stride = 1,
                batchSize = 2,
            )

        val batches = loader.batches()

        // samples = 4 -> batches [2, 2]
        assertThat(batches).hasSize(2)
        assertThat(batches[0]).hasSize(2)
        assertThat(batches[1]).hasSize(2)
    }

    @Test
    fun `batches keeps last partial batch`() {
        val loader =
            TextDataLoader(
                tokenIds = listOf(0, 1, 2, 3, 4),
                contextLength = 2,
                stride = 1,
                batchSize = 3,
            )

        val batches = loader.batches()

        // samples = 3 -> one full batch
        assertThat(batches).hasSize(1)
        assertThat(batches[0]).hasSize(3)
    }

    @Test
    fun `samples creates exactly one window when tokenIds size is contextLength plus one`() {
        val loader =
            TextDataLoader(
                tokenIds = listOf(10, 11, 12),
                contextLength = 2,
                stride = 1,
                batchSize = 1,
            )

        val samples = loader.samples()

        assertThat(samples).hasSize(1)
        assertThat(samples[0].inputIds).isEqualTo(listOf(10, 11))
        assertThat(samples[0].targetIds).isEqualTo(listOf(11, 12))
        assertThat(loader.size()).isEqualTo(1)
    }

    @Test
    fun `constructor rejects invalid parameters`() {
        assertThatThrownBy {
            TextDataLoader(
                tokenIds = listOf(1, 2, 3),
                contextLength = 0,
                stride = 1,
                batchSize = 1,
            )
        }.isInstanceOf(IllegalArgumentException::class.java)

        assertThatThrownBy {
            TextDataLoader(
                tokenIds = listOf(1, 2, 3),
                contextLength = 2,
                stride = 0,
                batchSize = 1,
            )
        }.isInstanceOf(IllegalArgumentException::class.java)

        assertThatThrownBy {
            TextDataLoader(
                tokenIds = listOf(1, 2, 3),
                contextLength = 2,
                stride = 1,
                batchSize = 0,
            )
        }.isInstanceOf(IllegalArgumentException::class.java)

        assertThatThrownBy {
            TextDataLoader(
                tokenIds = listOf(1, 2),
                contextLength = 2,
                stride = 1,
                batchSize = 1,
            )
        }.isInstanceOf(IllegalArgumentException::class.java)
            .hasMessage("tokenIds.size muss größer als contextLength sein")
    }
}
