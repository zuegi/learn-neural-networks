package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.library.GPT2Tokenizer
import ch.zuegi.ml.llm.shared.readVerdictText

/**
 * Der BPE Tokenizer GPT2Tokenizer kann mit jedem unbekannten Wort umgehen
 */
fun main() {
    val rawText = readVerdictText()
    val tokenizer = GPT2Tokenizer()
    val tokenIds = tokenizer.encode(rawText)

    println("Token-Anzahl: ${tokenIds.size}")

    // Schneide aus dem Text 12 tokenIds heraus
    val start = 350
    val length = 12
    val slice = tokenIds.drop(start).take(length)

    println("pos=position in text, id=token id")
    slice.forEachIndexed { i, id ->
        val pos = start + i
        val token = tokenizer.decode(listOf(id))
        println("pos=$pos, id=$id -> '$token'")
    }
    val randomTokenId: Int = tokenIds.get(357)
    val list2 = listOf(randomTokenId)
    val decodedTokenId: String = tokenizer.decode(list2)
    println("-> decoded token id: $randomTokenId -> $decodedTokenId")
    println("***")

    val text2 = "Hello, do you like tea? <|endoftext|> In the sunlit   terraces of someunknownPlace"
    val idList: List<Int> = tokenizer.encodeOrdinary(text2)
    println(idList)
    println(tokenizer.decode(idList))
    println("***")

    val text3 = "Akwirw ier"
    val idList3: List<Int> = tokenizer.encodeOrdinary(text3)
    println(idList3)
    println(tokenizer.decode(idList3))
    println("***")
}
