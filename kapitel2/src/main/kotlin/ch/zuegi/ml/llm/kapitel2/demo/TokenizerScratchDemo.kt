package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.scratch.SimpleTokenizerV1

fun main() {
    proveRoundtripWithKnownTokens()
    println()
    println("*** unknown token mapping ***")
    val rawText = "Darling, do you like water?"
    println("RawText (Vokabular-Basis): $rawText")
    val tokenizer = SimpleTokenizerV1(rawText)
    val text = "Hello, do you like tea?"
    println("Text zum Encodieren: $text")
    val idList: List<Int> = tokenizer.encode(text)
    println("idList.size: ${idList.size}")
    println("idList: $idList")
    println("Vokabular.size: ${tokenizer.vocabSize}")
    println("Vokabular: ${tokenizer.vocab}")
    println("decoded: ${tokenizer.decode(idList)}")
}

private fun proveRoundtripWithKnownTokens() {
    println("*** roundtrip with known tokens ***")
    val rawText = "Darling, do you like water?"
    val tokenizer = SimpleTokenizerV1(rawText)
    val text = "Darling, do you like water?"

    val ids = tokenizer.encode(text)
    val decoded = tokenizer.decode(ids)
    val isEqual = decoded == text

    println("text: $text")
    println("ids: $ids")
    println("decoded: $decoded")
    println("roundtrip.equal: $isEqual")

    check(isEqual) { "Roundtrip fehlgeschlagen: '$text' -> '$decoded'" }
}

