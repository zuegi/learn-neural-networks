package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.shared.tokenize.GPT2Tokenizer

/**
 * Der BPE-Tokenizer kann unbekannte Wörter über Subwords zerlegen und wieder verlustfrei zusammensetzen.
 */
fun main() {
    val tokenizer = GPT2Tokenizer()
    proveRoundtripWithKnownText(tokenizer)
    println()
    proveSpecialTokenAsPlainText(tokenizer)
    println()
    proveUnknownWordsUseSubwords(tokenizer)
}

private fun proveRoundtripWithKnownText(tokenizer: GPT2Tokenizer) {
    println("*** roundtrip with known text ***")
    val text = "Hello, do you like tea?"
    val ids = tokenizer.encode(text)
    val decoded = tokenizer.decode(ids)

    println("text: $text")
    println("ids: $ids")
    println("decoded: $decoded")
    println("roundtrip.equal: ${decoded == text}")

    check(ids.isNotEmpty()) { "Encoding darf nicht leer sein" }
    check(decoded == text) { "Roundtrip fehlgeschlagen: '$text' -> '$decoded'" }
}

private fun proveSpecialTokenAsPlainText(tokenizer: GPT2Tokenizer) {
    println("*** special token as plain text ***")
    val text = "Hello, do you like tea? <|endoftext|>"
    val ids = tokenizer.encodeOrdinary(text)
    val decoded = tokenizer.decode(ids)

    println("text: $text")
    println("ids: $ids")
    println("decoded: $decoded")
    println("roundtrip.equal: ${decoded == text}")

    check(decoded == text) { "Special-Token-Text wurde nicht verlustfrei rekonstruiert" }
}

private fun proveUnknownWordsUseSubwords(tokenizer: GPT2Tokenizer) {
    println("*** unknown words use subwords ***")
    val text = "Akwirw ier someunknownPlace"
    val ids = tokenizer.encodeOrdinary(text)
    val decoded = tokenizer.decode(ids)

    println("text: $text")
    println("ids: $ids")
    println("decoded: $decoded")
    println("roundtrip.equal: ${decoded == text}")
    println("tokenCount: ${ids.size}")

    check(ids.isNotEmpty()) { "Encoding darf nicht leer sein" }
    check(decoded == text) { "Unbekannte Wörter wurden nicht korrekt rekonstruiert" }
}
