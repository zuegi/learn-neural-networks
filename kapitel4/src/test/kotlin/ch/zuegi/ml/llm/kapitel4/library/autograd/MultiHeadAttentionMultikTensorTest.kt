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
import kotlin.math.max

class MultiHeadAttentionMultikTensorTest {
    @Test
    fun `softmax summiert zeilenweise auf eins und maskiert negative unendlich`() {
        val scores = TensorMultik(mk.ndarray(doubleArrayOf(1000.0, 999.0, 998.0)))
        val maskedScores = TensorMultik(mk.ndarray(doubleArrayOf(1.0, Double.NEGATIVE_INFINITY)))

        val weights = scores.softmax()
        val maskedWeights = maskedScores.softmax()

        assertEquals(1.0, (0 until weights.size).sumOf { weights.data[it] }, 1e-12)
        assertEquals(1.0, (0 until maskedWeights.size).sumOf { maskedWeights.data[it] }, 1e-12)
        assertEquals(0.0, maskedWeights.data[1], 0.0)
    }

    @Test
    fun `softmax bleibt bei grossen scores und kausaler maske endlich`() {
        val scores = TensorMultik(mk.ndarray(doubleArrayOf(1000.0, 999.0, 998.0)))
        val maskedScores = scores.maskCausalScale(position = 0, scale = 1.0)
        val weights = maskedScores.softmax()

        for (i in 0 until weights.size) {
            assertTrue(weights.data[i].isFinite())
        }
        assertEquals(1.0, weights.data[0], 0.0)
        assertEquals(0.0, weights.data[1], 0.0)
        assertEquals(0.0, weights.data[2], 0.0)
    }

    @Test
    fun `causal mask fuer ctx zwei sperrt nur zukunft`() {
        val firstPosition = TensorMultik(mk.ndarray(doubleArrayOf(1.0, 2.0)))
        val secondPosition = TensorMultik(mk.ndarray(doubleArrayOf(1.0, 2.0)))

        val firstWeights = firstPosition.maskCausalScale(position = 0, scale = 1.0).softmax()
        val secondWeights = secondPosition.maskCausalScale(position = 1, scale = 1.0).softmax()

        assertEquals(1.0, (0 until firstWeights.size).sumOf { firstWeights.data[it] }, 1e-12)
        assertEquals(0.0, firstWeights.data[1], 0.0)
        assertEquals(1.0, (0 until secondWeights.size).sumOf { secondWeights.data[it] }, 1e-12)
        assertTrue(secondWeights.data[0] > 0.0)
        assertTrue(secondWeights.data[1] > 0.0)
    }

    @Test
    fun `causal forward unterstuetzt ctx eins`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 8,
                numHeads = 2,
                dK = 4,
                causal = true,
                seed = 42,
            )

        val out = attention.forward(matrixInput(ctx = 1, dim = 8), ctx = 1, training = false)

        assertEquals(8, out.size)
    }

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

    // tag::library-mha-example[]
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
    // end::library-mha-example[]

    @Test
    fun `backward gradienten stimmen fuer ausgewaehlte parameter mit finite differences ueberein`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 4,
                numHeads = 2,
                dK = 2,
                useQkvBias = true,
                useOutputBias = true,
                seed = 31,
            )
        val input = matrixInput(ctx = 2, dim = 4)
        val parameters = listOf(
            attention.wQuery to 0,
            attention.wKey to 3,
            attention.wValue to 5,
            attention.wOutput to 2,
            checkNotNull(attention.bQuery) to 1,
            checkNotNull(attention.bOutput) to 2,
        )

        attention.parameters().forEach { it.zeroGrad() }
        objective(attention, input).backward()

        parameters.forEach { (parameter, index) ->
            val analytical = parameter.grad[index]
            val numerical = finiteDifference(attention, input, parameter, index)
            val tolerance = 1e-6 * max(1.0, max(abs(analytical), abs(numerical)))

            assertEquals(analytical, numerical, tolerance, "gradient mismatch at parameter index $index")
        }
    }

    @Test
    fun `embeddingDim kann sich von numHeads times dK unterscheiden`() {
        val attention =
            MultiHeadAttentionMultikTensor(
                embeddingDim = 10,
                numHeads = 2,
                dK = 4,
                useQkvBias = true,
                useOutputBias = true,
                seed = 13,
            )
        val input = matrixInput(ctx = 3, dim = 10)

        val out = attention.forward(input, ctx = 3, training = false)

        assertEquals(3 * 10, out.size)
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

    private fun objective(
        attention: MultiHeadAttentionMultikTensor,
        input: TensorMultik,
    ): TensorMultik = attention.forward(input, ctx = 2, training = false)

    private fun finiteDifference(
        attention: MultiHeadAttentionMultikTensor,
        input: TensorMultik,
        parameter: TensorMultik,
        index: Int,
    ): Double {
        val epsilon = 1e-6
        val original = parameter.data[index]

        parameter.data[index] = original + epsilon
        val plusOutput = objective(attention, input)
        val plus = (0 until plusOutput.size).sumOf { plusOutput.data[it] }
        parameter.data[index] = original - epsilon
        val minusOutput = objective(attention, input)
        val minus = (0 until minusOutput.size).sumOf { minusOutput.data[it] }
        parameter.data[index] = original

        return (plus - minus) / (2.0 * epsilon)
    }
}
