package ch.zuegi.ml.llm

class SimpleTokenizerV1(
    private val rawText: String,
) {
    private val tokenRegex = Regex("""\p{L}+(?:['’\-]\p{L}+)*|[.,!?;:"()]""")
    private var tokens: List<String> = emptyList()
    private var vocab: Map<String, Int> = emptyMap()
    private var idToToken: Map<Int, String> = emptyMap()

    init {
        tokens = tokenize(rawText)
        vocab = buildTokenToId(tokens)
        idToToken = vocab.entries.associate { (token, id) -> id to token }
    }

    fun encode(text: String): List<Int> =
        tokenize(text).map { token ->
            vocab[token] ?: error("Token nicht im Vokabular: '$token'")
        }

    fun decode(ids: List<Int>): String {
        val tokens =
            ids.map { id ->
                idToToken[id] ?: error("Unbekannte Token-ID: $id")
            }
        return joinTokens(tokens)
    }

    private fun joinTokens(tokens: List<String>): String {
        if (tokens.isEmpty()) return ""

        val noSpaceBefore = setOf(".", ",", "!", "?", ";", ":", ")", "]", "}", "\"")
        val noSpaceAfter = setOf("(", "[", "{")

        val out = StringBuilder()
        var quoteIsOpen = false

        for (token in tokens) {
            if (out.isEmpty()) {
                out.append(token)
                if (token == "\"") quoteIsOpen = true
                continue
            }

            val prevToken = tokenAtEnd(out)

            val needsSpace =
                when {
                    token == "\"" -> !quoteIsOpen && prevToken !in noSpaceAfter

                    // öffnendes Quote meist mit Space davor
                    token in noSpaceBefore -> false

                    prevToken in noSpaceAfter -> false

                    prevToken == "\"" && quoteIsOpen -> false

                    // direkt nach öffnendem Quote kein Space
                    else -> true
                }

            if (needsSpace) out.append(' ')
            out.append(token)

            if (token == "\"") {
                quoteIsOpen = !quoteIsOpen
            }
        }

        return out.toString()
    }

    private fun tokenAtEnd(out: StringBuilder): String {
        // Wir brauchen nur den letzten sichtbaren "Token-Anker" für Spacing-Regeln.
        return out.last().toString()
    }

    private fun tokenize(rawText: String): List<String> = tokenRegex.findAll(rawText).map { it.value }.toList()

    private fun buildTokenToId(tokens: List<String>): Map<String, Int> {
        val tokenToId = LinkedHashMap<String, Int>() // Reihenfolge stabil
        for (token in tokens) {
            if (token !in tokenToId) {
                tokenToId[token] = tokenToId.size
            }
        }
        return tokenToId
    }
}
