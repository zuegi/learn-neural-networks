package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.util.Random

class EmbeddingMultikTensorTest {
    private val embedding = EmbeddingMultikTensor(NUM_EMBEDDINGS, EMBEDDING_DIM, INIT_SCALE, Random(SEED))

    @Test
    fun `forward liefert die Zeile index der Tabelle`() {
        val out = embedding.forward(INDEX)

        assertEquals(EMBEDDING_DIM, out.size)
        for (k in 0 until EMBEDDING_DIM) {
            assertEquals(embedding.weight.data[INDEX * EMBEDDING_DIM + k], out.data[k])
        }
    }

    @Test
    fun `backward schreibt Gradient nur in die gelesene Zeile`() {
        embedding.forward(INDEX).backward()

        for (i in 0 until NUM_EMBEDDINGS * EMBEDDING_DIM) {
            val expected = if (i / EMBEDDING_DIM == INDEX) 1.0 else 0.0
            assertEquals(expected, embedding.weight.grad[i])
        }
    }

    @Test
    fun `forward lehnt index ausserhalb der Tabelle ab`() {
        assertThrows<IllegalArgumentException> { embedding.forward(NUM_EMBEDDINGS) }
    }

    private companion object {
        const val NUM_EMBEDDINGS = 5
        const val EMBEDDING_DIM = 3
        const val INIT_SCALE = 0.01
        const val SEED = 1L
        const val INDEX = 2
    }
}
