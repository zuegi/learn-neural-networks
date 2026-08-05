package ch.zuegi.ml.llm.autograd

import kotlin.math.tanh

/**
 * Autograd-Knoten fuer 1D-Vektoren (Etappe 1 Richtung Tensor-Autograd).
 *
 * Wie [Value], aber [data] und [grad] sind ganze Vektoren (`DoubleArray`).
 * Elementweise Operationen erzeugen neue Knoten und merken sich ihre lokale
 * Rueckwaertsregel. backward() propagiert den Gradienten in topologischer
 * Reihenfolge durch den Graphen; Gradienten werden elementweise akkumuliert.
 *
 * @param data Werte des Vektors.
 * @param children Eltern-Knoten, aus denen dieser Vektor entstanden ist.
 */
class Tensor(
    val data: DoubleArray,
    private val children: List<Tensor> = emptyList(),
) {
    val grad: DoubleArray = DoubleArray(data.size)

    private var backwardStep: () -> Unit = {}

    val size: Int get() = data.size

    /**
     * Elementweise Addition. Gradient fliesst 1:1 an beide Operanden:
     * d(a+b)/da = 1, d(a+b)/db = 1 (pro Element).
     */
    operator fun plus(other: Tensor): Tensor {
        require(size == other.size) { "Groessen muessen uebereinstimmen: $size vs ${other.size}" }
        val out = Tensor(DoubleArray(size) { data[it] + other.data[it] }, listOf(this, other))
        out.backwardStep = {
            for (i in 0 until size) {
                grad[i] += out.grad[i]
                other.grad[i] += out.grad[i]
            }
        }
        return out
    }

    /**
     * Elementweise Multiplikation (Hadamard). Produktregel pro Element:
     * d(a*b)/da = b, d(a*b)/db = a.
     */
    operator fun times(other: Tensor): Tensor {
        require(size == other.size) { "Groessen muessen uebereinstimmen: $size vs ${other.size}" }
        val out = Tensor(DoubleArray(size) { data[it] * other.data[it] }, listOf(this, other))
        out.backwardStep = {
            for (i in 0 until size) {
                grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += data[i] * out.grad[i]
            }
        }
        return out
    }

    /**
     * Elementweise tanh-Aktivierung. Ableitung pro Element: 1 - tanh(x)^2.
     */
    fun tanh(): Tensor {
        val t = DoubleArray(size) { tanh(data[it]) }
        val out = Tensor(t, listOf(this))
        out.backwardStep = {
            for (i in 0 until size) {
                grad[i] += (1.0 - t[i] * t[i]) * out.grad[i]
            }
        }
        return out
    }

    /**
     * Matrix-Vektor-Multiplikation: `y = weight · this`.
     *
     * this ist der Eingabevektor der Laenge n, [weight] eine Matrix [m, n].
     * Ergebnis ist ein Vektor der Laenge m.
     *
     * Backward-Regeln (y = W · x):
     * - dx = Wᵀ · dy   (Gradient an den Eingabevektor)
     * - dW = dy ⊗ x    (Gradient an die Gewichtsmatrix, aeusseres Produkt)
     *
     * @param weight lernbare Gewichtsmatrix als Tensor mit Form [m, n],
     *   flach gespeichert (row-major) in `weight.data`.
     * @param m Anzahl Ausgabezeilen.
     * @param n Anzahl Eingabespalten, muss `size` entsprechen.
     * @return Ergebnis-Tensor der Laenge m.
     */
    fun matVecMul(
        weight: Tensor,
        m: Int,
        n: Int,
    ): Tensor {
        require(size == n) { "Eingabegroesse $size passt nicht zu n=$n" }
        require(weight.size == m * n) { "weight.size ${weight.size} passt nicht zu m*n=${m * n}" }

        val result = DoubleArray(m)
        for (row in 0 until m) {
            var sum = 0.0
            val base = row * n
            for (col in 0 until n) {
                sum += weight.data[base + col] * data[col]
            }
            result[row] = sum
        }

        val out = Tensor(result, listOf(this, weight))
        out.backwardStep = {
            // dx = Wᵀ · dy
            for (col in 0 until n) {
                var sum = 0.0
                for (row in 0 until m) {
                    sum += weight.data[row * n + col] * out.grad[row]
                }
                grad[col] += sum
            }
            // dW = dy ⊗ x
            for (row in 0 until m) {
                val g = out.grad[row]
                val base = row * n
                for (col in 0 until n) {
                    weight.grad[base + col] += g * data[col]
                }
            }
        }
        return out
    }

    /**
     * Matrix-Matrix-Multiplikation: `C = this · other`.
     *
     * this ist Matrix A [p, q], [other] ist Matrix B [q, r], flach row-major
     * gespeichert. Ergebnis C ist [p, r].
     *
     * Backward-Regeln (C = A · B):
     * - dA = dC · Bᵀ   (Gradient an die linke Matrix)
     * - dB = Aᵀ · dC   (Gradient an die rechte Matrix)
     *
     * @param other rechte Matrix B, Form [q, r], flach in `other.data`.
     * @param p Zeilen von A.
     * @param q Spalten von A / Zeilen von B (innere Dimension).
     * @param r Spalten von B.
     * @return Ergebnis-Tensor der Groesse p * r (Form [p, r], row-major).
     */
    fun matMul(
        other: Tensor,
        p: Int,
        q: Int,
        r: Int,
    ): Tensor {
        require(size == p * q) { "this.size $size passt nicht zu p*q=${p * q}" }
        require(other.size == q * r) { "other.size ${other.size} passt nicht zu q*r=${q * r}" }

        val result = DoubleArray(p * r)
        for (i in 0 until p) {
            for (j in 0 until r) {
                var sum = 0.0
                for (k in 0 until q) {
                    sum += data[i * q + k] * other.data[k * r + j]
                }
                result[i * r + j] = sum
            }
        }

        val out = Tensor(result, listOf(this, other))
        out.backwardStep = {
            // dA = dC · Bᵀ  ->  dA[i,k] = Σ_j dC[i,j] * B[k,j]
            for (i in 0 until p) {
                for (k in 0 until q) {
                    var sum = 0.0
                    for (j in 0 until r) {
                        sum += out.grad[i * r + j] * other.data[k * r + j]
                    }
                    grad[i * q + k] += sum
                }
            }
            // dB = Aᵀ · dC  ->  dB[k,j] = Σ_i A[i,k] * dC[i,j]
            for (k in 0 until q) {
                for (j in 0 until r) {
                    var sum = 0.0
                    for (i in 0 until p) {
                        sum += data[i * q + k] * out.grad[i * r + j]
                    }
                    other.grad[k * r + j] += sum
                }
            }
        }
        return out
    }

    /**
     * Cross-Entropy-Loss ueber Softmax, kombiniert als eine Operation.
     *
     * this sind die Logits (Laenge vocabSize) fuer eine Position. Intern wird
     * softmax berechnet und der negative Log der Wahrscheinlichkeit der
     * Zielklasse [target] als Loss zurueckgegeben (ein Skalar-Tensor).
     *
     * Der kombinierte Backward ist besonders einfach:
     *   dLogits[i] = softmax(logits)[i] - (i == target ? 1 : 0)
     *
     * Das umgeht die volle Softmax-Jacobian.
     *
     * @param target Index der korrekten Klasse (0 until size).
     * @return Skalar-Tensor mit dem Loss.
     */
    fun softmaxCrossEntropy(target: Int): Tensor {
        require(target in 0 until size) { "target $target ausserhalb 0 until $size" }

        // numerisch stabiler softmax: max abziehen
        val max = data.max()
        val exps = DoubleArray(size) { kotlin.math.exp(data[it] - max) }
        val sumExp = exps.sum()
        val probs = DoubleArray(size) { exps[it] / sumExp }

        val loss = -kotlin.math.ln(probs[target])
        val out = Tensor(doubleArrayOf(loss), listOf(this))

        out.backwardStep = {
            // dLogits[i] = p[i] - oneHot[i], skaliert mit eingehendem grad
            val g = out.grad[0]
            for (i in 0 until size) {
                val oneHot = if (i == target) 1.0 else 0.0
                grad[i] += (probs[i] - oneHot) * g
            }
        }
        return out
    }

    /**
     * Rueckwaertsdurchlauf. Setzt grad dieses Knotens auf 1 (pro Element) und
     * propagiert in topologischer Reihenfolge rueckwaerts durch den Graphen.
     */
    fun backward() {
        val ordered = mutableListOf<Tensor>()
        val visited = mutableSetOf<Tensor>()

        fun buildTopo(node: Tensor) {
            if (node !in visited) {
                visited.add(node)
                node.children.forEach { buildTopo(it) }
                ordered.add(node)
            }
        }
        buildTopo(this)

        for (i in grad.indices) {
            grad[i] = 1.0
        }
        ordered.asReversed().forEach { it.backwardStep() }
    }
}
