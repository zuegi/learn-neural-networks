package ch.zuegi.ml.llm

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class SimpleTokenizerV1Test {
    private val trainingText =
        """
        "It's the last painted," you know.
        """.trimIndent()

    @Test
    fun `encode returns ids for known tokens`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val ids = tokenizer.encode("It's the last painted,")

        assertThat(ids).isNotEmpty
        assertThat(ids).allMatch { it >= 0 }
    }

    @Test
    fun `encode throws for unknown token`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        assertThatThrownBy { tokenizer.encode("UNKNOWN_TOKEN") }
            .isInstanceOf(IllegalStateException::class.java)
            .hasMessageContaining("Token nicht im Vokabular")
    }

    @Test
    fun `decode reconstructs punctuation spacing`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val ids = tokenizer.encode("\"It's the last painted,\" you know.")
        val decoded = tokenizer.decode(ids)

        assertThat(decoded).isEqualTo("\"It's the last painted,\" you know.")
    }

    @Test
    fun `encode decode roundtrip stays stable`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val original = "\"It's the last painted,\" you know."
        val decoded = tokenizer.decode(tokenizer.encode(original))

        assertThat(decoded).isEqualTo(original)
    }
}
