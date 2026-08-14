package ch.zuegi.ml.llm.kapitel5.training

import ch.zuegi.ml.llm.kapitel5.model.GPTModelMultikTensor
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.file.Files
import java.nio.file.Path

private const val CHECKPOINT_VERSION = 1

fun saveModelWeights(
    model: GPTModelMultikTensor,
    path: Path,
) {
    path.parent?.let { Files.createDirectories(it) }

    val parameters = model.parameters()
    DataOutputStream(Files.newOutputStream(path).buffered()).use { out ->
        out.writeInt(CHECKPOINT_VERSION)
        out.writeInt(parameters.size)
        for (parameter in parameters) {
            out.writeInt(parameter.size)
            for (i in 0 until parameter.size) {
                out.writeDouble(parameter.data[i])
            }
        }
    }
}

fun loadModelWeights(
    model: GPTModelMultikTensor,
    path: Path,
) {
    val parameters = model.parameters()

    DataInputStream(Files.newInputStream(path).buffered()).use { input ->
        val version = input.readInt()
        require(version == CHECKPOINT_VERSION) {
            "Checkpoint-Version $version wird nicht unterstuetzt"
        }

        val parameterCount = input.readInt()
        require(parameterCount == parameters.size) {
            "Checkpoint hat $parameterCount Parameter, Modell erwartet ${parameters.size}"
        }

        for (parameter in parameters) {
            val size = input.readInt()
            require(size == parameter.size) {
                "Checkpoint-Tensorgroesse $size passt nicht zu Modellgroesse ${parameter.size}"
            }

            for (i in 0 until parameter.size) {
                parameter.data[i] = input.readDouble()
            }
        }
    }
}

