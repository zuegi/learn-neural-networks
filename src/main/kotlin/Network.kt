import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import java.util.Random
import kotlin.math.exp

class Network(
    val sizes: IntArray,
) {
    /**
     * Python	                        Kotlin	                                Bedeutung
     * ======                           ======                                  =========
     * self.num_layers = len(sizes)	    val numLayers = sizes.size	            Wie viele Schichten hat das Netz
     * sizes[1:]	                    sizes.drop(1)	                        Alle Schichten außer der Eingabeschicht
     * sizes[:-1]	                    sizes.dropLast(1)	                    Alle Schichten außer der Ausgabeschicht
     * np.random.randn(y, 1)	        DoubleArray(y) { rnd.nextGaussian() }	Ein Bias pro Neuron, zufällig initialisiert
     * np.random.randn(y, x)	        Array(y) { DoubleArray(x) {...} }	    Eine Gewichts-Matrix zwischen zwei Schichten
     */

    private val rnd = Random()

    // Anzahl der Netwzerk Layers, z.B. [784, 30, 10] -> 3
    val numLayers: Int = sizes.size

    // biases[i] = Spaltenvektor für Schicht i+1 (die Eingabeschicht hat keinen Bias)
    // entspricht: [np.random.randn(y, 1) for y in sizes[1:]]
    var biases: List<DoubleArray> =
        sizes.drop(1).map { y ->
            DoubleArray(y) { rnd.nextGaussian() }
        }

    // weights[i] = Matrix zwischen Schicht i und Schicht i+1
    // entspricht: [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    var weights: List<Array<DoubleArray>> =
        sizes.dropLast(1).zip(sizes.drop(1)).map { (x, y) ->
            Array(y) { DoubleArray(x) { rnd.nextGaussian() } }
        }

    fun feedforward(input: DoubleArray): DoubleArray {
        var a = input
        for (i in biases.indices) {
            val z = vecAdd(matVecMul(weights[i], a), biases[i])
            a = sigmoidVector(z)
        }
        return a
    }

    suspend fun sgdParallel(
        trainingData: List<Pair<DoubleArray, DoubleArray>>,
        epochs: Int,
        miniBatchSize: Int,
        eta: Double,
        testData: List<Pair<DoubleArray, Int>>? = null,
        workers: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1),
    ) = coroutineScope {
        val shuffled = trainingData.toMutableList()
        val dispatcher = Dispatchers.Default.limitedParallelism(workers)

        for (j in 0 until epochs) {
            shuffled.shuffle(rnd)

            val miniBatches = shuffled.chunked(miniBatchSize)
            for (miniBatch in miniBatches) {
                updateMiniBatchParallel(miniBatch, eta, dispatcher, workers)
            }

            if (testData != null) {
                val correct = evaluate(testData)
                println("Epoch $j : $correct / ${testData.size}")
            } else {
                println("Epoch $j complete")
            }
        }
    }

    private suspend fun updateMiniBatchParallel(
        miniBatch: List<Pair<DoubleArray, DoubleArray>>,
        eta: Double,
        dispatcher: CoroutineDispatcher,
        workers: Int,
    ) = coroutineScope {
        val nablaB = biases.map { DoubleArray(it.size) }
        val nablaW = weights.map { w -> Array(w.size) { DoubleArray(w[0].size) } }

        val effectiveWorkers = minOf(workers.coerceAtLeast(1), miniBatch.size)
        val chunkSize = (miniBatch.size + effectiveWorkers - 1) / effectiveWorkers
        val chunks = miniBatch.chunked(chunkSize)

        val partials =
            chunks
                .map { chunk ->
                    async(Dispatchers.Default) {
                        val localB = biases.map { DoubleArray(it.size) }
                        val localW = weights.map { w -> Array(w.size) { DoubleArray(w[0].size) } }

                        for ((x, y) in chunk) {
                            val (db, dw) = backprop(x, y)

                            for (i in localB.indices) {
                                val lb = localB[i]
                                val dbi = db[i]
                                var k = 0
                                while (k < lb.size) {
                                    lb[k] += dbi[k]
                                    k++
                                }
                            }

                            for (i in localW.indices) {
                                val lw = localW[i]
                                val dwi = dw[i]
                                var r = 0
                                while (r < lw.size) {
                                    val lwr = lw[r]
                                    val dwr = dwi[r]
                                    var c = 0
                                    while (c < lwr.size) {
                                        lwr[c] += dwr[c]
                                        c++
                                    }
                                    r++
                                }
                            }
                        }

                        Pair(localB, localW)
                    }
                }.awaitAll()

        for ((pb, pw) in partials) {
            for (i in nablaB.indices) {
                val nb = nablaB[i]
                val pbi = pb[i]
                var k = 0
                while (k < nb.size) {
                    nb[k] += pbi[k]
                    k++
                }
            }

            for (i in nablaW.indices) {
                val nw = nablaW[i]
                val pwi = pw[i]
                var r = 0
                while (r < nw.size) {
                    val nwr = nw[r]
                    val pwr = pwi[r]
                    var c = 0
                    while (c < nwr.size) {
                        nwr[c] += pwr[c]
                        c++
                    }
                    r++
                }
            }
        }

        val scale = eta / miniBatch.size

        for (i in weights.indices) {
            val w = weights[i]
            val nw = nablaW[i]
            var r = 0
            while (r < w.size) {
                val wr = w[r]
                val nwr = nw[r]
                var c = 0
                while (c < wr.size) {
                    wr[c] -= scale * nwr[c]
                    c++
                }
                r++
            }
        }

        for (i in biases.indices) {
            val b = biases[i]
            val nb = nablaB[i]
            var k = 0
            while (k < b.size) {
                b[k] -= scale * nb[k]
                k++
            }
        }
    }

    /**
     * Die Kernidee von Backpropagation:
     * Wir rechnen den Fehler erst im Output aus ("wie falsch war die Vorhersage?"),
     * und schieben diesen Fehler dann rückwärts durch das Netz,
     * Schicht für Schicht – jede Schicht bekommt dabei ihren eigenen "Anteil" am Gesamtfehler zugewiesen.
     * Genau diese nablaB/nablaW-Werte werden dann in updateMiniBatch benutzt, um die Gewichte in die richtige Richtung zu verschieben.
     */
    fun backprop(
        x: DoubleArray,
        y: DoubleArray,
    ): Pair<List<DoubleArray>, List<Array<DoubleArray>>> {
        var activation = x
        val activations = mutableListOf(x)
        val zs = mutableListOf<DoubleArray>()

        for (i in biases.indices) {
            val z = vecAdd(matVecMul(weights[i], activation), biases[i])
            zs.add(z)
            activation = sigmoidVector(z)
            activations.add(activation)
        }

        val nablaB = MutableList(biases.size) { DoubleArray(biases[it].size) }
        val nablaW =
            MutableList(weights.size) { i ->
                Array(weights[i].size) { row -> DoubleArray(weights[i][row].size) }
            }

        val last = biases.size - 1

        var delta = vecSub(activations[last + 1], y)
        val spLast = sigmoidPrimeVector(zs[last])
        hadamardInPlace(delta, spLast)

        nablaB[last] = delta
        addOuterProductInPlace(nablaW[last], delta, activations[last])

        for (l in last - 1 downTo 0) {
            delta = transposeMatVecMul(weights[l + 1], delta)
            val sp = sigmoidPrimeVector(zs[l])
            hadamardInPlace(delta, sp)

            nablaB[l] = delta
            addOuterProductInPlace(nablaW[l], delta, activations[l])
        }

        return Pair(nablaB, nablaW)
    }

    // entspricht: def evaluate(self, test_data)
    // testData: Liste aus (Eingabebild, richtige Ziffer als Zahl 0-9)
    fun evaluate(testData: List<Pair<DoubleArray, Int>>): Int {
        val testResults =
            testData.map { (x, y) ->
                Pair(argmax(feedforward(x)), y)
            }
        return testResults.count { (predicted, actual) -> predicted == actual }
    }
}
