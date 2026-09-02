package ch.zuegi.ml.llm.kapitel4.scratch.autograd

import kotlin.math.pow
import kotlin.math.tanh


/**
 * Skalar * Value. Kotlin-Aequivalent zu Pythons `__rmul__`.
 * Ermoeglicht `3.0 * x`, indem der Skalar in ein Blatt-[Value] gewickelt wird.
 *
 * Das sind Kotlin Extension Functions
 */
@Suppress("ktlint:standard:no-consecutive-comments")
// tag::value-scalar-overloads-top-level[]
operator fun Double.times(value: Value): Value = Value(this) * value

/**
 * Skalar + Value. Kotlin-Aequivalent zu Pythons `__radd__`.
 * Ermoeglicht `3.0 + x`.
 */
operator fun Double.plus(value: Value): Value = Value(this) + value

/**
 * Skalar / Value. Kotlin-Aequivalent zu Pythons `__rtruediv__`.
 * Umgesetzt als `Value(skalar) * value^(-1)`, damit der Gradient ueber die
 * bestehenden `times`- und `pow`-Regeln automatisch korrekt fliesst.
 */
operator fun Double.div(value: Value): Value = Value(this) * value.pow(-1.0)
// end::value-scalar-overloads-top-level[]

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
@Suppress("ktlint:standard:no-consecutive-comments")
// tag::value-structure[]
class Value(
    var data: Double,
    private val children: List<Value> = emptyList(),
) {

    var grad: Double = 0.0

    // lokale Rueckwaertsregel dieser Operation, Default: nichts tun (Blatt)
    private var backwardStep: () -> Unit = {}
    // end::value-structure[]



    /** Addition zweier Values. Gradient fliesst 1:1 an beide Operanden (d(a+b)=1). */
    // tag::value-plus[]
    operator fun plus(other: Value): Value {
        val out = Value(data + other.data, listOf(this, other))
        out.backwardStep = {
            // d(a+b)/da = 1, d(a+b)/db = 1
            grad += out.grad
            other.grad += out.grad
        }
        return out
    }
    // end::value-plus[]

    /** Multiplikation zweier Values. Produktregel: d(a*b)/da = b, d(a*b)/db = a. */
    // tag::value-times[]
    operator fun times(other: Value): Value {
        val out = Value(data * other.data, listOf(this, other))
        out.backwardStep = {
            // Produktregel: d(a*b)/da = b, d(a*b)/db = a
            grad += other.data * out.grad
            other.grad += data * out.grad
        }
        return out
    }
    // end::value-times[]

    /** tanh-Aktivierung. Ableitung: 1 - tanh(x)². */
    // tag::value-tanh[]
    fun tanh(): Value {
        val t = tanh(data)
        val out = Value(t, listOf(this))
        out.backwardStep = {
            // d(tanh(x))/dx = 1 - tanh(x)^2
            grad += (1.0 - t * t) * out.grad
        }
        return out
    }
    // end::value-tanh[]

    /** Potenz mit konstantem Exponenten. Ableitung: n * x^(n-1). */
    // tag::value-pow[]
    fun pow(exponent: Double): Value {
        val out = Value(data.pow(exponent), listOf(this))
        out.backwardStep = {
            // d(x^n)/dx = n * x^(n-1)
            grad += exponent * data.pow(exponent - 1) * out.grad
        }
        return out
    }
    // end::value-pow[]

    // tag::value-scalar-overloads-member[]

    /**
     * Value * Skalar, z.B. `x * 3.0`. Wickelt den Skalar in ein Blatt-[Value].
     */
    operator fun times(scalar: Double): Value = this * Value(scalar)

    /**
     * Value + Skalar, z.B. `x + 3.0`.
     */
    operator fun plus(scalar: Double): Value = this + Value(scalar)
    // end::value-scalar-overloads-member[]


    /**
     * Value / Value, umgesetzt als `this * other^(-1)`.
     * Die Quotientenregel entsteht automatisch aus `times`- und `pow`-Backward;
     * beide Operanden erhalten korrekt ihren Gradienten (auch `other` via `-a/b²`).
     */
    // tag::value-div[]
    operator fun div(other: Value): Value = this * other.pow(-1.0)

    operator fun div(scalar: Double): Value = this * Value(scalar).pow(-1.0)
    // end::value-div[]



    /**
     * Rueckwaertsdurchlauf: setzt den Gradienten dieses Knotens auf 1 und
     * propagiert ihn in topologischer Reihenfolge durch den ganzen Graphen.
     */
    // tag::value-backward[]
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
    // end::value-backward[]
}
