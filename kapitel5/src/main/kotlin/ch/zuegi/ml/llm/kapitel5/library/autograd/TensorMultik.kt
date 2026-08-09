package ch.zuegi.ml.llm.kapitel5.library.autograd

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import java.util.Collections.emptyList

class TensorMultik(
    val data: NDArray<Double, D1>,
    private val children: List<TensorMultik> = emptyList(),
) {
    val grad: NDArray<Double, D1> = mk.ndarray(DoubleArray(data.size) { 0.0 })
    var backwardStep: () -> Unit = {}
    val size: Int get() = data.size

    companion object {
        fun stackRows(
            rows: List<TensorMultik>,
            len: Int,
        ): TensorMultik {
            require(rows.all { it.size == len }) { "alle Zeilen muessen Laenge $len haben" }
            val result = DoubleArray(rows.size * len)
            for (i in rows.indices) {
                for (k in 0 until len) {
                    result[i * len + k] = rows[i].data[k]
                }
            }
            val out = TensorMultik(mk.ndarray(result), rows)
            out.backwardStep = {
                for (i in rows.indices) {
                    for (k in 0 until len) {
                        rows[i].grad[k] += out.grad[i * len + k]
                    }
                }
            }
            return out
        }

        fun concatCols(
            mats: List<TensorMultik>,
            ctx: Int,
            colsEach: Int,
        ): TensorMultik {
            require(mats.all { it.size == ctx * colsEach }) {
                "jede Matrix muss ctx*colsEach=${ctx * colsEach} gross sein"
            }
            val blocks = mats.size
            val totalCols = blocks * colsEach
            val result = DoubleArray(ctx * totalCols)
            for (row in 0 until ctx) {
                for (b in 0 until blocks) {
                    val src = mats[b]
                    for (c in 0 until colsEach) {
                        result[row * totalCols + b * colsEach + c] = src.data[row * colsEach + c]
                    }
                }
            }
            val out = TensorMultik(mk.ndarray(result), mats)
            out.backwardStep = {
                for (row in 0 until ctx) {
                    for (b in 0 until blocks) {
                        val src = mats[b]
                        for (c in 0 until colsEach) {
                            src.grad[row * colsEach + c] += out.grad[row * totalCols + b * colsEach + c]
                        }
                    }
                }
            }
            return out
        }
    }

    fun zeroGrad() {
        for (i in 0 until grad.size) grad[i] = 0.0
    }

    operator fun plus(other: TensorMultik): TensorMultik {
        require(size == other.size) { "Groessen muessen uebereinstimmen: $size vs ${other.size}" }
        val result = DoubleArray(size) { i -> data[i] + other.data[i] }
        val out = TensorMultik(mk.ndarray(result), listOf(this, other))
        out.backwardStep = {
            for (i in 0 until size) {
                grad[i] += out.grad[i]
                other.grad[i] += out.grad[i]
            }
        }
        return out
    }

    operator fun times(other: TensorMultik): TensorMultik {
        require(size == other.size) { "Groessen muessen uebereinstimmen: $size vs ${other.size}" }
        val result = DoubleArray(size) { i -> data[i] * other.data[i] }
        val out = TensorMultik(mk.ndarray(result), listOf(this, other))
        out.backwardStep = {
            for (i in 0 until size) {
                grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += data[i] * out.grad[i]
            }
        }
        return out
    }

    fun row(
        rowIndex: Int,
        len: Int,
    ): TensorMultik {
        require(rowIndex * len + len <= size) { "Zeile $rowIndex (len $len) ausserhalb size $size" }
        val slice = DoubleArray(len) { i -> data[rowIndex * len + i] }
        val out = TensorMultik(mk.ndarray(slice), listOf(this))
        out.backwardStep = {
            for (i in 0 until len) {
                grad[rowIndex * len + i] += out.grad[i]
            }
        }
        return out
    }

    fun dotWithRows(
        row: TensorMultik,
        rows: Int,
        cols: Int,
    ): TensorMultik {
        require(size == rows * cols) { "size $size passt nicht zu rows*cols=${rows * cols}" }
        require(row.size == cols) { "row.size ${row.size} passt nicht zu cols=$cols" }

        val result =
            DoubleArray(rows) { i ->
                var sum = 0.0
                for (c in 0 until cols) sum += data[i * cols + c] * row.data[c]
                sum
            }

        val out = TensorMultik(mk.ndarray(result), listOf(this, row))
        out.backwardStep = {
            for (i in 0 until rows) {
                val g = out.grad[i]
                for (c in 0 until cols) {
                    grad[i * cols + c] += row.data[c] * g
                    row.grad[c] += data[i * cols + c] * g
                }
            }
        }
        return out
    }

    fun matVecMul(
        weight: TensorMultik,
        m: Int,
        n: Int,
    ): TensorMultik {
        require(size == n) { "Eingabegroesse $size passt nicht zu n=$n" }
        require(weight.size == m * n) { "weight.size ${weight.size} passt nicht zu m*n=${m * n}" }
        val result = DoubleArray(m)
        for (row in 0 until m) {
            var sum = 0.0
            val base = row * n
            for (col in 0 until n) sum += weight.data[base + col] * data[col]
            result[row] = sum
        }
        val out = TensorMultik(mk.ndarray(result), listOf(this, weight))
        out.backwardStep = {
            for (col in 0 until n) {
                var sum = 0.0
                for (row in 0 until m) sum += weight.data[row * n + col] * out.grad[row]
                grad[col] += sum
            }
            for (row in 0 until m) {
                val g = out.grad[row]
                val base = row * n
                for (col in 0 until n) weight.grad[base + col] += g * data[col]
            }
        }
        return out
    }

    fun matMul(
        other: TensorMultik,
        p: Int,
        q: Int,
        r: Int,
    ): TensorMultik {
        require(size == p * q) { "this.size $size passt nicht zu p*q=${p * q}" }
        require(other.size == q * r) { "other.size ${other.size} passt nicht zu q*r=${q * r}" }
        val result = DoubleArray(p * r)
        for (i in 0 until p) {
            for (j in 0 until r) {
                var sum = 0.0
                for (k in 0 until q) sum += data[i * q + k] * other.data[k * r + j]
                result[i * r + j] = sum
            }
        }
        val out = TensorMultik(mk.ndarray(result), listOf(this, other))
        out.backwardStep = {
            for (i in 0 until p) {
                for (k in 0 until q) {
                    var sum = 0.0
                    for (j in 0 until r) sum += out.grad[i * r + j] * other.data[k * r + j]
                    grad[i * q + k] += sum
                }
            }
            for (k in 0 until q) {
                for (j in 0 until r) {
                    var sum = 0.0
                    for (i in 0 until p) sum += data[i * q + k] * out.grad[i * r + j]
                    other.grad[k * r + j] += sum
                }
            }
        }
        return out
    }

    fun backward() {
        val ordered = mutableListOf<TensorMultik>()
        val visited = mutableSetOf<TensorMultik>()

        fun buildTopo(node: TensorMultik) {
            if (node !in visited) {
                visited.add(node)
                node.children.forEach { buildTopo(it) }
                ordered.add(node)
            }
        }
        buildTopo(this)
        for (i in 0 until grad.size) grad[i] = 1.0
        ordered.asReversed().forEach { it.backwardStep() }
    }

    fun softmax(): TensorMultik {
        val max = (0 until size).maxOf { data[it] }
        val exps = DoubleArray(size) { i -> kotlin.math.exp(data[i] - max) }
        val sumExp = exps.sum()
        val probs = DoubleArray(size) { i -> exps[i] / sumExp }

        val out = TensorMultik(mk.ndarray(probs), listOf(this))
        out.backwardStep = {
            var dot = 0.0
            for (i in 0 until size) dot += probs[i] * out.grad[i]
            for (i in 0 until size) grad[i] += probs[i] * (out.grad[i] - dot)
        }
        return out
    }

    fun softmaxCrossEntropy(target: Int): TensorMultik {
        require(target in 0 until size) { "target $target ausserhalb 0 until $size" }

        val max = (0 until size).maxOf { data[it] }
        val exps = DoubleArray(size) { i -> kotlin.math.exp(data[i] - max) }
        val sumExp = exps.sum()
        val probs = DoubleArray(size) { i -> exps[i] / sumExp }

        val loss = -kotlin.math.ln(probs[target])
        val out = TensorMultik(mk.ndarray(doubleArrayOf(loss)), listOf(this))

        out.backwardStep = {
            val g = out.grad[0]
            for (i in 0 until size) {
                val oneHot = if (i == target) 1.0 else 0.0
                grad[i] += (probs[i] - oneHot) * g
            }
        }
        return out
    }

    fun gelu(): TensorMultik {
        val c = kotlin.math.sqrt(2.0 / kotlin.math.PI)
        val a = 0.044715

        val tValues = DoubleArray(size)
        val result =
            DoubleArray(size) { i ->
                val x = data[i]
                val inner = c * (x + a * x * x * x)
                val t = kotlin.math.tanh(inner)
                tValues[i] = t
                0.5 * x * (1.0 + t)
            }

        val out = TensorMultik(mk.ndarray(result), listOf(this))
        out.backwardStep = {
            for (i in 0 until size) {
                val x = data[i]
                val t = tValues[i]
                val dInner = c * (1.0 + 3.0 * a * x * x)
                val dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dInner
                grad[i] += dgelu * out.grad[i]
            }
        }
        return out
    }

    fun dropout(
        probability: Double,
        rnd: java.util.Random,
    ): TensorMultik {
        require(probability in 0.0..1.0) { "probability muss in [0.0, 1.0] liegen" }

        if (probability == 0.0) return this

        if (probability == 1.0) {
            val out = TensorMultik(mk.ndarray(DoubleArray(size)), listOf(this))
            out.backwardStep = {}
            return out
        }

        val keepScale = 1.0 / (1.0 - probability)
        val kept = BooleanArray(size)
        val result =
            DoubleArray(size) { i ->
                val keep = rnd.nextDouble() >= probability
                kept[i] = keep
                if (keep) data[i] * keepScale else 0.0
            }

        val out = TensorMultik(mk.ndarray(result), listOf(this))
        out.backwardStep = {
            for (i in 0 until size) {
                if (kept[i]) grad[i] += out.grad[i] * keepScale
            }
        }
        return out
    }

    fun scale(factor: Double): TensorMultik {
        val result = DoubleArray(size) { i -> data[i] / factor }
        val out = TensorMultik(mk.ndarray(result), listOf(this))
        out.backwardStep = {
            for (i in 0 until size) grad[i] += out.grad[i] / factor
        }
        return out
    }

    fun maskCausalScale(
        position: Int,
        scale: Double,
    ): TensorMultik {
        val result =
            DoubleArray(size) { j ->
                if (j > position) Double.NEGATIVE_INFINITY else data[j] / scale
            }
        val out = TensorMultik(mk.ndarray(result), listOf(this))
        out.backwardStep = {
            for (j in 0 until size) {
                if (j <= position) grad[j] += out.grad[j] / scale
            }
        }
        return out
    }

    fun layerNorm(
        gamma: TensorMultik,
        beta: TensorMultik,
        eps: Double = 1e-5,
    ): TensorMultik {
        require(gamma.size == size) { "gamma.size ${gamma.size} passt nicht zu $size" }
        require(beta.size == size) { "beta.size ${beta.size} passt nicht zu $size" }

        val n = size
        val mean = (0 until n).sumOf { data[it] } / n
        val variance =
            (0 until n).sumOf { i ->
                val d = data[i] - mean
                d * d
            } / n
        val std = kotlin.math.sqrt(variance + eps)

        val xhat = DoubleArray(n) { i -> (data[i] - mean) / std }
        val result = DoubleArray(n) { i -> gamma.data[i] * xhat[i] + beta.data[i] }

        val out = TensorMultik(mk.ndarray(result), listOf(this, gamma, beta))
        out.backwardStep = {
            val dxhat = DoubleArray(n) { i -> out.grad[i] * gamma.data[i] }
            var meanDxhat = 0.0
            var meanDxhatXhat = 0.0
            for (i in 0 until n) {
                meanDxhat += dxhat[i]
                meanDxhatXhat += dxhat[i] * xhat[i]
            }
            meanDxhat /= n
            meanDxhatXhat /= n

            for (i in 0 until n) {
                grad[i] += (dxhat[i] - meanDxhat - xhat[i] * meanDxhatXhat) / std
                gamma.grad[i] += out.grad[i] * xhat[i]
                beta.grad[i] += out.grad[i]
            }
        }
        return out
    }

    fun sliceCols(
        rows: Int,
        cols: Int,
        fromCol: Int,
        width: Int,
    ): TensorMultik {
        val result = DoubleArray(rows * width)
        for (r in 0 until rows) {
            for (c in 0 until width) {
                result[r * width + c] = data[r * cols + fromCol + c]
            }
        }
        val out = TensorMultik(mk.ndarray(result), listOf(this))
        out.backwardStep = {
            for (r in 0 until rows) {
                for (c in 0 until width) {
                    grad[r * cols + fromCol + c] += out.grad[r * width + c]
                }
            }
        }
        return out
    }

    fun broadcastMul(other: TensorMultik): TensorMultik {
        require(size == 1) { "broadcastMul erwartet linken Tensor der Laenge 1, war $size" }
        val scalar = data[0]
        val result = DoubleArray(other.size) { i -> scalar * other.data[i] }

        val out = TensorMultik(mk.ndarray(result), listOf(this, other))
        out.backwardStep = {
            var scalarGrad = 0.0
            for (i in 0 until other.size) {
                scalarGrad += other.data[i] * out.grad[i]
                other.grad[i] += scalar * out.grad[i]
            }
            grad[0] += scalarGrad
        }
        return out
    }

    fun addBias(
        rows: Int,
        cols: Int,
        bias: TensorMultik,
    ): TensorMultik {
        require(size == rows * cols) { "size $size passt nicht zu rows*cols=${rows * cols}" }
        require(bias.size == cols) { "bias.size ${bias.size} passt nicht zu cols=$cols" }

        val result = DoubleArray(size)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                result[r * cols + c] = data[r * cols + c] + bias.data[c]
            }
        }

        val out = TensorMultik(mk.ndarray(result), listOf(this, bias))
        out.backwardStep = {
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    val g = out.grad[r * cols + c]
                    grad[r * cols + c] += g
                    bias.grad[c] += g
                }
            }
        }
        return out
    }
}
