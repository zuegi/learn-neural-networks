package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class TokenEmbeddingTest {
    @Test
    fun `weights have shape vocabSize x embeddingDim`() {
        val embedding = TokenEmbedding(vocabSize = 10, embeddingDim = 4)

        assertThat(embedding.weights.size).isEqualTo(10)
        assertThat(embedding.weights.all { it.size == 4 }).isTrue()
    }

    @Test
    fun `lookup returns embedding row for token id`() {
        val embedding = TokenEmbedding(vocabSize = 5, embeddingDim = 3)

        val vector = embedding.lookup(2)

        assertThat(vector).isSameAs(embedding.weights[2])
        assertThat(vector.size).isEqualTo(3)
    }

    @Test
    fun `lookup of sequence returns matrix with embedding per token`() {
        val embedding = TokenEmbedding(vocabSize = 5, embeddingDim = 3)

        val matrix = embedding.lookup(listOf(0, 4, 1))

        assertThat(matrix.size).isEqualTo(3)
        assertThat(matrix[0]).isSameAs(embedding.weights[0])
        assertThat(matrix[1]).isSameAs(embedding.weights[4])
        assertThat(matrix[2]).isSameAs(embedding.weights[1])
    }

    @Test
    fun `same seed produces same weights`() {
        val a = TokenEmbedding(vocabSize = 6, embeddingDim = 4, seed = 42)
        val b = TokenEmbedding(vocabSize = 6, embeddingDim = 4, seed = 42)

        assertThat(a.weights.contentDeepEquals(b.weights)).isTrue()
    }

    @Test
    fun `lookup throws for token id out of range`() {
        val embedding = TokenEmbedding(vocabSize = 3, embeddingDim = 2)

        assertThatThrownBy { embedding.lookup(3) }
            .isInstanceOf(IllegalArgumentException::class.java)
    }

    @Test
    fun `constructor rejects invalid dimensions`() {
        assertThatThrownBy { TokenEmbedding(vocabSize = 0, embeddingDim = 4) }
            .isInstanceOf(IllegalArgumentException::class.java)

        assertThatThrownBy { TokenEmbedding(vocabSize = 4, embeddingDim = 0) }
            .isInstanceOf(IllegalArgumentException::class.java)
    }
}

