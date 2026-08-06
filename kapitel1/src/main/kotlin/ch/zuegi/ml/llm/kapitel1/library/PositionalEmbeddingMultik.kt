package ch.zuegi.ml.llm.kapitel1.library

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import java.util.Random

class PositionalEmbeddingMultik(
    private val contextLength: Int,
    private val embeddingDim: Int,
    seed: Long? = null,
) {
    init {
        require(contextLength > 0) { "contextLength muss > 0 sein" }
        require(embeddingDim > 0) { "embeddingDim muss > 0 sein" }
    }

    private val rnd = if (seed != null) Random(seed) else Random()

    val weights: NDArray<Double, D2> =
        mk.ndarray(
            List(contextLength) {
                List(embeddingDim) { rnd.nextGaussian() * INIT_SCALE }
            },
        )

    fun lookup(position: Int): MultiArray<Double, D1> {
        require(position in 0 until contextLength) { "position $position ausserhalb 0 until $contextLength" }
        return weights[position]
    }

    fun lookupAll(): NDArray<Double, D2> = weights

    companion object {
        private const val INIT_SCALE = 0.01
    }
}
