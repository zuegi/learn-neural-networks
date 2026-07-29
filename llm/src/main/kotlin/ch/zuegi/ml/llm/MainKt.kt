package ch.zuegi.ml.llm

import java.awt.SystemColor.text
import java.nio.charset.StandardCharsets

// Wörter + interne Apostrophe/Bindestriche, Satzzeichen separat als Tokens

fun main() {
    val rawText = readVerdictText()
    val text =
        """
        "It's the last painted, you know",
        Mrs. Gisburn said with pardonable pride.
        """.trimIndent()

    println("Zeichen: ${text.length}")

    val tokenizer1 = SimpleTokenizerV1(rawText)
    val ids: List<Int> = tokenizer1.encode(text)
    println(ids)
    println(tokenizer1.decode(ids))

    // Fehler fall
    val text2 = "Hello, do you like tea?"
    val tokenizer2 = SimpleTokenizerV1(rawText)
    val idList: List<Int> = tokenizer2.encode(text2)
    println(idList)
    println(tokenizer1.decode(idList))
}

fun readVerdictText(): String {
    val resourcePath = "/text/the-verdict.txt"
    val stream =
        object {}.javaClass.getResourceAsStream(resourcePath)
            ?: error("Ressource nicht gefunden: $resourcePath")

    return stream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
}
