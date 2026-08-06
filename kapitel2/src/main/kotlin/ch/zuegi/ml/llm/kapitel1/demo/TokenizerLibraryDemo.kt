package ch.zuegi.ml.llm.kapitel1.demo

import ch.zuegi.ml.llm.shared.readVerdictText
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList

/**
 * Der BPE Tokenizer kann mit jedem unbekannten Wort umgehen
 */
fun main() {
    // 1. Registry laden
    val registry = Encodings.newDefaultEncodingRegistry()

    // 2. GPT-2 Encoding (r50k_base) abrufen
    val tokenizer = registry.getEncoding(EncodingType.R50K_BASE)

    // 3. Text codieren
    val rawText = readVerdictText()
    val tokenIds = tokenizer.encode(rawText).toArray().toList()

    // 4. Ergebnis ausgeben
    println("Token-Anzahl: ${tokenIds.size}")

    // Schneide aus dem Text 12 tokenIds heraus
    val start = 350
    val length = 12
    val slice = tokenIds.drop(start).take(length)

    println("pos=position in text, id=token id")
    slice.forEachIndexed { i, id ->
        val pos = start + i
        val tokenList = IntArrayList().also { it.add(id) }
        val token = tokenizer.decode(tokenList)
        println("pos=$pos, id=$id -> '$token'")
    }
    val randomTokenId: Int = tokenIds.get(357)
    val list2 = listOf(randomTokenId).toIntArrayList()
    val decodedTokenId: String = tokenizer.decode(list2)
    println("-> decoded token id: $randomTokenId -> $decodedTokenId")
    println("***")

    /*
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
    val text2 = "Hello, do you like tea? <|endoftext|> In the sunlit   terraces of someunknownPlace"
    val idList: List<Int> = tokenizer.encodeOrdinary(text2).boxed().toMutableList()
    println(idList)
    println(tokenizer.decode(idList.toIntArrayList()))
    println("***")

    val text3 = "Akwirw ier"
    val idList3: List<Int> = tokenizer.encodeOrdinary(text3).boxed().toMutableList()
    println(idList3)
    println(tokenizer.decode(idList3.toIntArrayList()))
    println("***")
}

fun List<Int>.toIntArrayList(): IntArrayList = IntArrayList(this.size).also { out -> this.forEach(out::add) }
