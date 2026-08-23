package ch.zuegi.ml.llm.kapitel2.library

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.IntArrayList
import com.knuddels.jtokkit.api.ModelType

class R50kBpeTokenizer {
    private val encoding: Encoding =
        Encodings
            .newDefaultEncodingRegistry()
            .getEncodingForModel(ModelType.ADA)

    val vocabSize: Int = 50257 // Die Vokabular Size die das ADA Modell definiert

    fun encode(text: String): List<Int> = encoding.encode(text).toArray().toList()

    fun decode(tokenIds: List<Int>): String = encoding.decode(tokenIds.toIntArray())

    /**
     * Encodiert Text ohne Special-Token-Parsing.
     * Spezialsequenzen wie `<|endoftext|>` werden als normaler Text behandelt.
     */
    fun encodeOrdinary(text: String): List<Int> = encoding.encodeOrdinary(text).boxed().toList()
}

private fun List<Int>.toIntArray(): IntArrayList = IntArrayList(this.size).also { out -> this.forEach(out::add) }
