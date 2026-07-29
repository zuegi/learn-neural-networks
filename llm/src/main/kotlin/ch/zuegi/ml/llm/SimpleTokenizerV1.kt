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
        val punctuationNoSpaceBefore = setOf(".", ",", "!", "?", ";", ":", ")", "]", "}")
        val openingBrackets = setOf("(", "[", "{")
        val out = StringBuilder()

        var quoteOpen = false

        for (id in ids) {
            val token = idToToken[id] ?: error("Unbekannte Token-ID: $id")

            if (out.isEmpty()) {
                out.append(token)
                if (token == "\"") quoteOpen = !quoteOpen
                continue
            }

            val prev = out.last().toString()

            when {
                token == "\"" -> {
                    if (quoteOpen) {
                        // schliessendes Quote: kein Leerzeichen davor
                        out.append(token)
                    } else {
                        // öffnendes Quote: i.d.R. mit Leerzeichen davor, ausser nach öffnender Klammer
                        if (prev in openingBrackets) out.append(token) else out.append(' ').append(token)
                    }
                    quoteOpen = !quoteOpen
                }

                token in punctuationNoSpaceBefore -> {
                    out.append(token)
                }

                prev in openingBrackets || prev == "\"" -> {
                    out.append(token)
                }

                else -> {
                    out.append(' ').append(token)
                }
            }
        }

        return out.toString()
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
