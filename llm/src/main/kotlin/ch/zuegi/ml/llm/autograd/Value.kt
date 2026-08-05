package ch.zuegi.ml.llm.autograd

import kotlin.math.tanh

/**
 * Skalarer Autograd-Knoten (Micrograd-Stil). Siehe auch https://www.youtube.com/watch?v=VMj-3S1tku0
 *
 * Jeder Value speichert einen Zahlenwert und seinen Gradienten. Operationen
 * erzeugen neue Value-Knoten und merken sich, wie der Gradient rueckwaerts an
 * die Eltern fliesst (lokale Ableitung). backward() rechnet die Kettenregel
 * ueber den gesamten Rechengraphen rueckwaerts.
 *
 * @param data der Zahlenwert dieses Knotens.
 * @param children Eltern-Knoten, aus denen dieser Wert entstanden ist.
 */
class Value(
    var data: Double,
    private val children: List<Value> = emptyList(),
) {
    var grad: Double = 0.0

    // lokale Rueckwaertsregel dieser Operation, Default: nichts tun (Blatt)
    private var backwardStep: () -> Unit = {}

    operator fun plus(other: Value): Value {
        val out = Value(data + other.data, listOf(this, other))
        out.backwardStep = {
            // d(a+b)/da = 1, d(a+b)/db = 1
            grad += out.grad
            other.grad += out.grad
        }
        return out
    }

    operator fun times(other: Value): Value {
        val out = Value(data * other.data, listOf(this, other))
        out.backwardStep = {
            // Produktregel: d(a*b)/da = b, d(a*b)/db = a
            grad += other.data * out.grad
            other.grad += data * out.grad
        }
        return out
    }

    fun tanh(): Value {
        val t = tanh(data)
        val out = Value(t, listOf(this))
        out.backwardStep = {
            // d(tanh(x))/dx = 1 - tanh(x)^2
            grad += (1.0 - t * t) * out.grad
        }
        return out
    }

    /**
     * Rueckwaertsdurchlauf: setzt den Gradienten dieses Knotens auf 1 und
     * propagiert ihn in topologischer Reihenfolge durch den ganzen Graphen.
     */
    fun backward() {
        val ordered = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()

        fun buildTopo(node: Value) {
            if (node !in visited) {
                visited.add(node)
                node.children.forEach { buildTopo(it) }
                ordered.add(node)
            }
        }
        buildTopo(this)

        grad = 1.0
        ordered.asReversed().forEach { it.backwardStep() }
    }
}
