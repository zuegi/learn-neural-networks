package ch.zuegi.ml.llm

import ch.zuegi.ml.llm.SimpleTokenizerV1.Companion.ENDOFTEXT
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class SimpleTokenizerV1Test {
    private val trainingText = "\"It's the last painted,\" you know."

    @Test
    fun `encode returns ids for known tokens`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val ids = tokenizer.encode("\"It's the last painted,\"")

        assertThat(ids).isNotEmpty
        assertThat(ids).allMatch { it >= 0 }
    }

    @Test
    fun `vocab contains special tokens with stable ids at end`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val unkId = tokenizer.encode("<|unk|>")[0]
        val eotId = tokenizer.encode("<|endoftext|>")[0]

        assertThat(unkId).isGreaterThanOrEqualTo(0)
        assertThat(eotId).isEqualTo(unkId + 1)
    }

    @Test
    fun `encode maps unknown token to unk id`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val unkId = tokenizer.encode("<|unk|>")[0]
        val encoded = tokenizer.encode("UNKNOWN_TOKEN")[0]

        assertThat(encoded).isEqualTo(unkId)
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

    @Test
    fun `encode keeps explicit endoftext token`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val eotId = tokenizer.encode(ENDOFTEXT).single()
        val encoded = tokenizer.encode("you know.$ENDOFTEXT")

        assertThat(encoded).endsWith(eotId)
    }

    @Test
    fun `decode keeps explicit endoftext token without extra blank`() {
        val tokenizer = SimpleTokenizerV1(trainingText)

        val text = "you know.$ENDOFTEXT"
        val decoded = tokenizer.decode(tokenizer.encode(text))

        assertThat(decoded).isEqualTo(text)
    }
}
