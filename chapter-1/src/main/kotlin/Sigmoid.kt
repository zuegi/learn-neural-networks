import kotlin.math.exp

// Sigmoid für eine einzelne Zahl
fun sigmoid(z: Double): Double = 1.0 / (1.0 + exp(-z))

// Sigmoid für einen ganzen Vektor (Spalte von Zahlen),
// entspricht np.exp(-z) auf einem ganzen Array
fun sigmoidVector(z: DoubleArray): DoubleArray = DoubleArray(z.size) { i -> sigmoid(z[i]) }

// entspricht sigmoid_prime(z) = sigmoid(z)*(1-sigmoid(z))
fun sigmoidPrimeVector(z: DoubleArray): DoubleArray =
    DoubleArray(z.size) { i ->
        val s = sigmoid(z[i])
        s * (1 - s)
    }
