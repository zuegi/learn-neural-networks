package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.scratch.SimpleTokenizerV1
import ch.zuegi.ml.llm.shared.readVerdictText

fun main() {
    val rawText = readVerdictText()
    val tokenizer = SimpleTokenizerV1(rawText)
    val tokenIds = tokenizer.encode(rawText)

    println("Anzahl token ids: ${tokenIds.size}")

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
    val decodedTokenId: String = tokenizer.decode(listOf(randomTokenId))
    println("-> decoded token id: $randomTokenId -> $decodedTokenId")
    println("***")

    val text3 = "Akwirw ier"
    val idList3: List<Int> = tokenizer.encode(text3)
    println(idList3)
    println(tokenizer.decode(idList3))
    println("***")

    // Fehler Fall: Token nicht im Vokabular: 'Hello' und wird mit <|unk|> ersetzt
    tokenNotInVocabular()
}

private fun tokenNotInVocabular() {
    println("*** token not in vocabular ***")
    val rawText = "Darling, do you like water?"
    println("Vokabular: $rawText")
    val tokenizer = SimpleTokenizerV1(rawText)
    val text = "Hello, do you like tea?"
    println("Text zu encode: $text")
    val idList: List<Int> = tokenizer.encode(text)
    println("idList.size: ${idList.size}")
    println("Vokabular.size: ${tokenizer.vocabSize}")
    println("Vokabular: ${tokenizer.vocab}")
    println(tokenizer.decode(idList))

}
