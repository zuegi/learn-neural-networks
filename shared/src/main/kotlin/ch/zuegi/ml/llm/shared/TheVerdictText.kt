package ch.zuegi.ml.llm.shared

import java.nio.charset.StandardCharsets

data class TheVerdictText(
    val text: String,
) {
    fun corpusSizeInCharacters(): Int = text.length
}

fun readTheVerdictText(): TheVerdictText = TheVerdictText(loadVerdictText())

fun readVerdictText(): String {
    return loadVerdictText()
}

private fun loadVerdictText(): String {
    val resourcePath = "/text/the-verdict.txt"
    val stream =
        object {}.javaClass.getResourceAsStream(resourcePath)
            ?: error("Ressource nicht gefunden: $resourcePath")

    return stream.bufferedReader(StandardCharsets.UTF_8).use { it.readText() }
}
