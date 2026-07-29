import jdk.internal.vm.vector.VectorSupport.test
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import kotlin.time.measureTime

suspend fun main() {
    val dauer =
        measureTime {
            /**
             * the IntArray sizes contains the number of neurons in the respective layers.
             * 784 neurons in Input Layer
             * 30 neurons in Hidden Layer
             * 10 neurons in Output Layer
             */
            val sizes: IntArray = intArrayOf(784, 200, 100, 30, 10)
            val epochs = 30
            val miniBatchSize = 64
            val workers = 8 // cores
            val learningRate = 6.0

            val net = Network(sizes)
            println("Anzahl Layers: ${net.numLayers}")
            println("Bias-Vektoren pro Layer: ${net.biases.map { it.size }}") // [30, 10]
            println("Gewichts-Matrizen (Zeilen x Spalten): ${net.weights.map { "${it.size}x${it[0].size}" }}") // [30x784, 10x30]

            val trainImages = loadImages("/mnist/train-images-idx3-ubyte")
            val trainLabels = loadLabels("/mnist/train-labels-idx1-ubyte")
            val testImages = loadImages("/mnist/t10k-images-idx3-ubyte")
            val testLabels = loadLabels("/mnist/t10k-labels-idx1-ubyte")

            // Trainingsdaten: (Bild, one-hot-Vektor) — passend zu sgd(...)
            val trainingData =
                trainImages.zip(trainLabels) { image, label ->
                    Pair(image, oneHot(label))
                }

            // Testdaten: (Bild, Ziffer als Zahl) — passend zu evaluate(...)
            val testData = testImages.zip(testLabels)

            // Trainiere
            // TODO: Wofür stehen die Variablen
            net.sgdParallel(
                trainingData,
                epochs = epochs,
                miniBatchSize = miniBatchSize,
                eta = learningRate,
                testData = testData,
                workers = workers,
            )
        }

    println("Dauer: $dauer")
}
