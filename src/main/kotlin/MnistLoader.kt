import java.io.BufferedInputStream
import java.io.DataInputStream

// Liest eine IDX-Bild-Datei und gibt eine Liste von Pixel-Vektoren zurück
// (jeder Vektor: 784 Werte zwischen 0.0 und 1.0, entspricht np.reshape(784,1)/255 im Original)
fun loadImages(resourcePath: String): List<DoubleArray> {
    val stream =
        object {}.javaClass.getResourceAsStream(resourcePath)
            ?: error("Ressource nicht gefunden: $resourcePath")

    DataInputStream(BufferedInputStream(stream)).use { input ->
        val magicNumber = input.readInt()
        require(magicNumber == 2051) { "Keine gültige MNIST-Bild-Datei: $resourcePath" }

        val numImages = input.readInt()
        val numRows = input.readInt()
        val numCols = input.readInt()
        val imageSize = numRows * numCols

        return List(numImages) {
            val pixels = ByteArray(imageSize)
            input.readFully(pixels)
            DoubleArray(imageSize) { i -> (pixels[i].toInt() and 0xFF) / 255.0 }
        }
    }
}

// Liest eine IDX-Label-Datei und gibt eine Liste von Ziffern (0-9) zurück
fun loadLabels(resourcePath: String): List<Int> {
    val stream =
        object {}.javaClass.getResourceAsStream(resourcePath)
            ?: error("Ressource nicht gefunden: $resourcePath")

    DataInputStream(BufferedInputStream(stream)).use { input ->
        val magicNumber = input.readInt()
        require(magicNumber == 2049) { "Keine gültige MNIST-Label-Datei: $resourcePath" }

        val numLabels = input.readInt()
        return List(numLabels) { input.readUnsignedByte() }
    }
}

// entspricht vectorized_result(j) aus Nielsens mnist_loader.py:
// wandelt eine Ziffer (z.B. 3) in einen "one-hot"-Vektor um: [0,0,0,1,0,0,0,0,0,0]
fun oneHot(label: Int): DoubleArray = DoubleArray(10) { i -> if (i == label) 1.0 else 0.0 }
