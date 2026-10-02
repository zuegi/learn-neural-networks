package ch.zuegi.ml.llm.kapitel5.training

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.DataOutputStream
import java.nio.file.Files
import java.nio.file.Path

class ModelCheckpointTest {
    @TempDir
    lateinit var tempDir: Path

    @Test
    fun `gespeicherte Gewichte werden in ein anders initialisiertes Modell geladen`() {
        val source = tinyModel(seed = 1L)
        val target = tinyModel(seed = 2L)
        val path = tempDir.resolve("checkpoints/model.bin")

        saveModelWeights(source, path)
        loadModelWeights(target, path)

        target.parameterSnapshot().zip(source.parameterSnapshot()).forEach { (loaded, saved) ->
            assertArrayEquals(saved, loaded)
        }
    }

    @Test
    fun `unbekannte Checkpoint-Version wird abgelehnt`() {
        val path = tempDir.resolve("invalid.bin")
        DataOutputStream(Files.newOutputStream(path)).use { it.writeInt(UNSUPPORTED_VERSION) }

        assertThrows(IllegalArgumentException::class.java) { loadModelWeights(tinyModel(), path) }
    }

    companion object {
        private const val UNSUPPORTED_VERSION = 99
    }
}
