package ch.zuegi.ml.llm.kapitel5.library.tokenize

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

    /**
     Encodes the given text into a list of token ids.
     Special tokens are artificial tokens used to unlock capabilities from a model, such as fill-in-the-middle.
     There is no support for parsing special tokens in a text, so if the text contains special tokens,
     this method will throw an UnsupportedOperationException.
     If you want to encode special tokens as ordinary text, use encodeOrdinary(String).
     Encoding encoding = EncodingRegistry.getEncoding(EncodingType.CL100K_BASE);
     encoding.encode("hello world");
     // returns [15339, 1917]
     encoding.encode("hello <|endoftext|> world");
     raises an UnsupportedOperationException
     */
    fun encodeOrdinary(text: String): List<Int> = encoding.encodeOrdinary(text).boxed().toMutableList()
}

private fun List<Int>.toIntArray(): IntArrayList = IntArrayList(this.size).also { out -> this.forEach(out::add) }
