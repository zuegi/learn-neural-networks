/*
 * NumKo ist eine Hilfsklasse, mit welcher Matrizen Berechnungen durchgeführt werden
 */

fun matVecMul(
    w: Array<DoubleArray>,
    a: DoubleArray,
): DoubleArray {
    val y = w.size
    val out = DoubleArray(y)
    var row = 0
    while (row < y) {
        val wr = w[row]
        var sum = 0.0
        var col = 0
        while (col < wr.size) {
            sum += wr[col] * a[col]
            col++
        }
        out[row] = sum
        row++
    }
    return out
}

// Cache-freundlich: zeilenweise über w laufen, Ergebnis akkumulieren.
fun transposeMatVecMul(
    w: Array<DoubleArray>,
    delta: DoubleArray,
): DoubleArray {
    val x = w[0].size
    val out = DoubleArray(x)
    var row = 0
    while (row < w.size) {
        val wr = w[row]
        val d = delta[row]
        var col = 0
        while (col < x) {
            out[col] += wr[col] * d
            col++
        }
        row++
    }
    return out
}

fun vecAdd(
    a: DoubleArray,
    b: DoubleArray,
): DoubleArray {
    val out = DoubleArray(a.size)
    var i = 0
    while (i < a.size) {
        out[i] = a[i] + b[i]
        i++
    }
    return out
}

fun vecSub(
    a: DoubleArray,
    b: DoubleArray,
): DoubleArray {
    val out = DoubleArray(a.size)
    var i = 0
    while (i < a.size) {
        out[i] = a[i] - b[i]
        i++
    }
    return out
}

fun hadamard(
    a: DoubleArray,
    b: DoubleArray,
): DoubleArray {
    val out = DoubleArray(a.size)
    var i = 0
    while (i < a.size) {
        out[i] = a[i] * b[i]
        i++
    }
    return out
}

// In-place Helfer, reduziert temporäre Arrays in Hot Paths.
fun addInPlace(
    target: DoubleArray,
    other: DoubleArray,
) {
    var i = 0
    while (i < target.size) {
        target[i] += other[i]
        i++
    }
}

fun subInPlaceScaled(
    target: DoubleArray,
    grad: DoubleArray,
    scale: Double,
) {
    var i = 0
    while (i < target.size) {
        target[i] -= scale * grad[i]
        i++
    }
}

fun hadamardInPlace(
    target: DoubleArray,
    other: DoubleArray,
) {
    var i = 0
    while (i < target.size) {
        target[i] *= other[i]
        i++
    }
}

// Akkumuliert outer product direkt in Zielmatrix (kein neues Array).
fun addOuterProductInPlace(
    target: Array<DoubleArray>,
    delta: DoubleArray,
    activation: DoubleArray,
) {
    var row = 0
    while (row < delta.size) {
        val tr = target[row]
        val d = delta[row]
        var col = 0
        while (col < activation.size) {
            tr[col] += d * activation[col]
            col++
        }
        row++
    }
}

fun argmax(a: DoubleArray): Int {
    var maxIndex = 0
    var i = 1
    while (i < a.size) {
        if (a[i] > a[maxIndex]) maxIndex = i
        i++
    }
    return maxIndex
}
