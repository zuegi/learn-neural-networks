package ch.zuegi.ml.llm.shared.tokenize

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList

class GPT2Tokenizer {
    private val encoding: Encoding =
        Encodings
            .newDefaultEncodingRegistry()
            .getEncoding(EncodingType.R50K_BASE)

    val vocabSize: Int = 50257

    fun encode(text: String): List<Int> = encoding.encode(text).toArray().toList()

    fun decode(tokenIds: List<Int>): String = encoding.decode(tokenIds.toIntArray())

    fun encodeOrdinary(text: String): List<Int> = encoding.encodeOrdinary(text).boxed().toMutableList()
}

private fun List<Int>.toIntArray(): IntArrayList = IntArrayList(this.size).also { out -> this.forEach(out::add) }

