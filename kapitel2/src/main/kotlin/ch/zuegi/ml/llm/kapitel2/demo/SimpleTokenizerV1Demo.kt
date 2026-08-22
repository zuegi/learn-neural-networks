package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.kapitel2.scratch.SimpleTokenizerV1

fun main() {
    proveRoundtripWithKnownTokens()
    println()
    proveUnknownTokenMapping()
}

private fun proveUnknownTokenMapping() {
    println("*** unknown token mapping ***")
    val rawText = "Darling, do you like water?"
    println("RawText (Vokabular-Basis): $rawText")
    val tokenizer = SimpleTokenizerV1(rawText)
    val text = "Hello, do you like tea?"
    println("Text zum Encodieren: $text")
    val idList: List<Int> = tokenizer.encode(text)
    val expectedIds = listOf(7, 1, 2, 3, 4, 7, 6)
    val expectedDecoded = "<|unk|>, do you like<|unk|>?"
    val decoded = tokenizer.decode(idList)

    println("idList.size: ${idList.size}")
    println("idList: $idList")
    println("Vokabular.size: ${tokenizer.vocabSize}")
    println("Vokabular: ${tokenizer.vocab}")
    println("decoded: $decoded")
    println("expectedIds: $expectedIds")
    println("expectedDecoded: $expectedDecoded")

    check(idList == expectedIds) { "Unerwartete Token-IDs: $idList" }
    check(decoded == expectedDecoded) { "Unerwartetes Decoding: '$decoded'" }
    check(tokenizer.vocabSize == 9) { "Unerwartete Vokabulargröße: ${tokenizer.vocabSize}" }
}

private fun proveRoundtripWithKnownTokens() {
    println("*** roundtrip with known tokens ***")
    val rawText = "Darling, do you like water?"
    val tokenizer = SimpleTokenizerV1(rawText)
    val text = "Darling, do you like water?"

    val ids = tokenizer.encode(text)
    val decoded = tokenizer.decode(ids)
    val expectedIds = listOf(0, 1, 2, 3, 4, 5, 6)
    val isEqual = decoded == text

    println("text: $text")
    println("ids: $ids")
    println("decoded: $decoded")
    println("expectedIds: $expectedIds")
    println("roundtrip.equal: $isEqual")

    check(ids == expectedIds) { "Unerwartete Token-IDs: $ids" }
    check(isEqual) { "Roundtrip fehlgeschlagen: '$text' -> '$decoded'" }
}

