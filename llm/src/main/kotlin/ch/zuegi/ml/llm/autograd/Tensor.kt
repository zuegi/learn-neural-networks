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

    companion object {
        fun stackRows(
            rows: List<Tensor>,
            len: Int,
        ): Tensor {
            require(rows.all { it.size == len }) { "alle Zeilen muessen Laenge $len haben" }
            val count = rows.size
            val result = DoubleArray(count * len)
            for (i in 0 until count) {
                for (k in 0 until len) result[i * len + k] = rows[i].data[k]
            }
            val out = Tensor(result, rows)
            out.backwardStep = {
                for (i in 0 until count) {
                    for (k in 0 until len) rows[i].grad[k] += out.grad[i * len + k]
                }
            }
            return out
        }

        /**
         * Konkateniert mehrere Matrix-Tensoren [ctx, colsEach] spaltenweise zu
         * einem [ctx, mats.size * colsEach]-Tensor (row-major).
         * Backward verteilt den Gradienten je Block zurueck an die Eltern.
         */
        fun concatCols(
            mats: List<Tensor>,
            ctx: Int,
            colsEach: Int,
        ): Tensor {
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
            val out = Tensor(result, mats)
            out.backwardStep = {
                for (row in 0 until ctx) {
                    for (b in 0 until blocks) {
                        val src = mats[b]
                        for (c in 0 until colsEach) {
                            src.grad[row * colsEach + c] +=
                                out.grad[row * totalCols + b * colsEach + c]
                        }
                    }
                }
            }
            return out
        }
    }

    /**
     * Setzt alle Gradienten dieses Knotens auf 0.
     *
     * Notwendig zwischen Trainings-Schritten, weil Gradienten in den
     * Backward-Regeln akkumuliert werden (`+=`). Ohne Zuruecksetzen wuerden
     * sich die Gradienten ueber mehrere Schritte aufsummieren.
     */
    fun zeroGrad() {
        grad.fill(0.0)
    }

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
     * Layer Normalization ueber diesen Vektor (eine Token-Zeile).
     *
     * Normalisiert `this` auf Mittelwert 0 und Varianz 1 und wendet die
     * lernbaren Parameter [gamma] (Skalierung) und [beta] (Verschiebung) an:
     *
     *     mean  = (1/n) Σ x_i
     *     var   = (1/n) Σ (x_i - mean)^2
     *     xhat  = (x - mean) / sqrt(var + eps)
     *     y     = gamma * xhat + beta
     *
     * Backward (Standard-LayerNorm-Gradient) fuer jede Komponente i:
     *
     *     dxhat_i = dy_i * gamma_i
     *     dx_i    = (1/std) * (dxhat_i - mean(dxhat) - xhat_i * mean(dxhat*xhat))
     *     dgamma_i = dy_i * xhat_i
     *     dbeta_i  = dy_i
     *
     * @param gamma Skalierungsparameter, Laenge n.
     * @param beta Verschiebungsparameter, Laenge n.
     * @param eps kleiner Wert fuer numerische Stabilitaet.
     * @return normalisierter und transformierter Tensor der Laenge n.
     */
    fun layerNorm(
        gamma: Tensor,
        beta: Tensor,
        eps: Double = 1e-5,
    ): Tensor {
        require(gamma.size == size) { "gamma.size ${gamma.size} passt nicht zu $size" }
        require(beta.size == size) { "beta.size ${beta.size} passt nicht zu $size" }

        val n = size
        val mean = data.average()
        val variance = data.sumOf { (it - mean) * (it - mean) } / n
        val std = kotlin.math.sqrt(variance + eps)

        val xhat = DoubleArray(n) { (data[it] - mean) / std }
        val result = DoubleArray(n) { gamma.data[it] * xhat[it] + beta.data[it] }

        val out = Tensor(result, listOf(this, gamma, beta))
        out.backwardStep = {
            // dgamma, dbeta direkt; dxhat = dy * gamma
            val dxhat = DoubleArray(n) { out.grad[it] * gamma.data[it] }
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

    /**
     * GELU-Aktivierung (tanh-Approximation, wie GPT-2), elementweise.
     *
     *     inner = sqrt(2/π) * (x + 0.044715 * x^3)
     *     gelu  = 0.5 * x * (1 + tanh(inner))
     *
     * Ableitung pro Element:
     *
     *     t          = tanh(inner)
     *     dInner/dx  = sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
     *     dgelu/dx   = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * dInner/dx
     *
     * @return Tensor gleicher Laenge mit GELU pro Element.
     */
    fun gelu(): Tensor {
        val c = kotlin.math.sqrt(2.0 / kotlin.math.PI)
        val a = 0.044715

        val tValues = DoubleArray(size)
        val result =
            DoubleArray(size) { i ->
                val x = data[i]
                val inner = c * (x + a * x * x * x)
                val t = tanh(inner)
                tValues[i] = t
                0.5 * x * (1.0 + t)
            }

        val out = Tensor(result, listOf(this))
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

    /**
     * Softmax ueber diesen Vektor. p_i = exp(x_i-max) / Σ exp(x_j-max).
     * Backward (volle Jacobian, kompakt): dx_i = p_i * (dy_i - Σ_j p_j*dy_j).
     */
    fun softmax(): Tensor {
        val max = data.max()
        val exps = DoubleArray(size) { kotlin.math.exp(data[it] - max) }
        val sumExp = exps.sum()
        val probs = DoubleArray(size) { exps[it] / sumExp }

        val out = Tensor(probs, listOf(this))
        out.backwardStep = {
            var dot = 0.0
            for (i in 0 until size) dot += probs[i] * out.grad[i]
            for (i in 0 until size) grad[i] += probs[i] * (out.grad[i] - dot)
        }
        return out
    }

    /**
     * Extrahiert Zeile [i] (Laenge [len]) aus diesem [rows, len]-Matrix-Tensor.
     * Backward leitet den Zeilen-Gradienten an die passenden Zeilen-Positionen.
     */
    fun row(
        i: Int,
        len: Int,
    ): Tensor {
        require(i * len + len <= size) { "Zeile $i (len $len) ausserhalb size $size" }
        val slice = DoubleArray(len) { data[i * len + it] }
        val out = Tensor(slice, listOf(this))
        out.backwardStep = {
            for (k in 0 until len) grad[i * len + k] += out.grad[k]
        }
        return out
    }

    /**
     * Transponiert diesen [rows, cols]-Matrix-Tensor zu [cols, rows].
     * Backward transponiert den Gradienten zurueck.
     */
    fun transposeMatrix(
        rows: Int,
        cols: Int,
    ): Tensor {
        require(size == rows * cols) { "size $size passt nicht zu rows*cols=${rows * cols}" }
        val result = DoubleArray(size)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                result[c * rows + r] = data[r * cols + c]
            }
        }
        val out = Tensor(result, listOf(this))
        out.backwardStep = {
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    grad[r * cols + c] += out.grad[c * rows + r]
                }
            }
        }
        return out
    }

    /**
     * Maskiert Zukunfts-Positionen (j > [position]) mit -inf und skaliert die
     * uebrigen mit 1/[scale]. Fuer Causal-Attention-Scores einer Zeile.
     * Backward: erlaubte Positionen bekommen grad/scale, maskierte 0.
     */
    fun maskCausalScale(
        position: Int,
        scale: Double,
    ): Tensor {
        val result =
            DoubleArray(size) { j ->
                if (j > position) Double.NEGATIVE_INFINITY else data[j] / scale
            }
        val out = Tensor(result, listOf(this))
        out.backwardStep = {
            for (j in 0 until size) {
                if (j <= position) grad[j] += out.grad[j] / scale
            }
        }
        return out
    }

    /**
     * Nur-Skalierung (nicht-causal): teilt alle Werte durch [scale].
     */
    fun scale(scale: Double): Tensor {
        val result = DoubleArray(size) { data[it] / scale }
        val out = Tensor(result, listOf(this))
        out.backwardStep = {
            for (j in 0 until size) grad[j] += out.grad[j] / scale
        }
        return out
    }

    /**
     * Embedding-Lookup: extrahiert Zeile [tokenId] (Laenge [dim]) aus dieser
     * [rows, dim]-Tabelle. Wie [row], aber semantisch als lernbare
     * Embedding-Tabelle gedacht. Backward schreibt den Zeilen-Gradienten
     * zurueck in die getroffene Tabellenzeile.
     *
     * @param tokenId Zeilenindex (Token- oder Positions-ID).
     * @param dim Laenge einer Tabellenzeile.
     */
    fun embeddingLookup(
        tokenId: Int,
        dim: Int,
    ): Tensor = row(tokenId, dim)

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
