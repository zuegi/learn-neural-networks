package ch.zuegi.ml.llm.shared.embedding

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import java.util.Random

class TokenEmbeddingMultik(
    val vocabSize: Int,
    private val embeddingDim: Int,
    seed: Long? = null,
) {
    init {
        require(vocabSize > 0) { "vocabSize muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val weights: NDArray<Double, D2> =
        mk.ndarray(
            List(vocabSize) {
                List(embeddingDim) { rnd.nextGaussian() * INIT_SCALE }
            },
        )

    fun lookup(tokenId: Int): MultiArray<Double, D1> {
        require(tokenId in 0 until vocabSize) { "tokenId $tokenId ausserhalb 0 until $vocabSize" }
        return weights[tokenId]
    }

    fun lookup(tokenIds: List<Int>): NDArray<Double, D2> {
        val rows = tokenIds.map { lookup(it).toList() }
        return mk.ndarray(rows)
    }

    companion object {
        private const val INIT_SCALE = 0.01
    }
}

