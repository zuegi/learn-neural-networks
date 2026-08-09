package ch.zuegi.ml.llm.kapitel4.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs

class MultiHeadAttentionMultikTensorTest {
    @Test
    fun `forward liefert output gleicher laenge wie input`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 42,
            )
        val input = matrixInput(ctx = 3, dim = 8)

        val out = attention.forward(input, ctx = 3, training = false)

        assertEquals(3 * 8, out.size)
    }

    @Test
    fun `gleiches seed erzeugt gleiche outputs auch mit bias`() {
        val a1 =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 7,
            )
        val a2 =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 7,
            )
        val input = matrixInput(ctx = 3, dim = 8)

        val out1 = a1.forward(input, ctx = 3, training = false)
        val out2 = a2.forward(input, ctx = 3, training = false)

        for (i in 0 until out1.size) {
            assertEquals(out1.data[i], out2.data[i], 1e-12)
        }
    }

    @Test
    fun `dropout im training beeinflusst output`() {
        val noDropout =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                dropoutProb = 0.0,
                useQkvBias = true,
                useOutputBias = true,
                seed = 11,
            )
        val withDropout =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                dropoutProb = 0.5,
                useQkvBias = true,
                useOutputBias = true,
                seed = 11,
            )
        val input = matrixInput(ctx = 3, dim = 8)

        val out1 = noDropout.forward(input, ctx = 3, training = true)
        val out2 = withDropout.forward(input, ctx = 3, training = true)

        val allEqual = (0 until out1.size).all { i -> abs(out1.data[i] - out2.data[i]) < 1e-12 }

        assertFalse(allEqual)
    }

    @Test
    fun `backward setzt gradienten auf gewichten und bias parametern`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 21,
            )
        val input = matrixInput(ctx = 3, dim = 8)

        val out = attention.forward(input, ctx = 3, training = false)
        out.backward()

        val hasGradient =
            attention.parameters().any { parameter ->
                (0 until parameter.size).any { i -> parameter.grad[i] != 0.0 }
            }

        assertTrue(hasGradient)

        val biasParameters = listOfNotNull(attention.bQuery, attention.bKey, attention.bValue, attention.bOutput)
        val biasHasGradient =
            biasParameters.all { parameter ->
                (0 until parameter.size).any { i -> parameter.grad[i] != 0.0 }
            }

        assertTrue(biasHasGradient)
    }

    @Test
    fun `parameter liste enthaelt bias tensoren wenn aktiviert`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 5,
            )

        assertEquals(8, attention.parameters().size)
    }

    @Test
    fun `parameter liste enthaelt nur gewichte wenn bias deaktiviert`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = false,
                useOutputBias = false,
                seed = 5,
            )

        assertEquals(4, attention.parameters().size)
    }

    @Test
    fun `bias initial auf null liefert gleiches output wie ohne bias`() {
        val withoutBias =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = false,
                useOutputBias = false,
                seed = 9,
            )
        val withBias =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 9,
            )
        val input = matrixInput(ctx = 3, dim = 8)

        val out1 = withoutBias.forward(input, ctx = 3, training = false)
        val out2 = withBias.forward(input, ctx = 3, training = false)

        for (i in 0 until out1.size) {
            assertEquals(out1.data[i], out2.data[i], 1e-12)
        }
    }

    @Test
    fun `gesetzter output bias veraendert output`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 9,
            )
        val input = matrixInput(ctx = 3, dim = 8)

        val outBefore = attention.forward(input, ctx = 3, training = false)

        for (i in 0 until attention.bOutput!!.size) {
            attention.bOutput.data[i] = 0.25
        }

        val outAfter = attention.forward(input, ctx = 3, training = false)

        val allEqual = (0 until outBefore.size).all { i -> abs(outBefore.data[i] - outAfter.data[i]) < 1e-12 }

        assertFalse(allEqual)
    }

    private fun matrixInput(
        ctx: Int,
        dim: Int,
    ): TensorMultik =
        TensorMultik(
            mk.ndarray(
                DoubleArray(ctx * dim) { index ->
                    val row = index / dim
                    val col = index % dim
                    if (row == col) 1.0 else (row + col) * 0.1
                },
            ),
        )
}
