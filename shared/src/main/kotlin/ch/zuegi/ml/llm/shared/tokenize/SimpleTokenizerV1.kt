package ch.zuegi.ml.llm.shared.tokenize

/**
 * Einfacher regelbasierter Tokenizer für Textdaten.
 *
 * Dieser Tokenizer ist nur für das Verständnis des Konzepts erforderlich.
 *
 * Der Tokenizer baut beim Erzeugen ein Vokabular aus dem übergebenen `rawText`.
 * Jedes unterschiedliche Token erhält dabei eine stabile numerische ID.
 *
 * Tokenisierung:
 * - erkennt Wörter inklusive Apostroph, typografischem Apostroph, Bindestrich und Unterstrich
 * - erkennt einfache Satzzeichen als eigene Tokens
 * - erkennt die Special Tokens [UNKNOWN] und [ENDOFTEXT]
 * - unterscheidet Groß- und Kleinschreibung
 *
 * Unbekannte Tokens werden beim Encodieren nicht verworfen und lösen keinen Fehler aus.
 * Stattdessen werden sie auf die ID von [UNKNOWN] gemappt.
 *
 * [ENDOFTEXT] wird nicht automatisch angehängt. Es erscheint nur dann im Encoding,
 * wenn es explizit im Eingabetext vorkommt.
 *
 * @param rawText Text, aus dem das Vokabular aufgebaut wird.
 */
class SimpleTokenizerV1(
    rawText: String,
) {
    private val tokenRegex = Regex("""<\|unk\|>|<\|endoftext\|>|\p{L}+(?:[_'’\-]\p{L}+)*|[.,!?;:\"()]""")
    private var tokens: List<String> = emptyList()
    private var vocab: Map<String, Int> = emptyMap()
    private var idToToken: Map<Int, String> = emptyMap()
    val vocabSize: Int get() = vocab.size

    companion object {
        const val UNKNOWN = "<|unk|>"
        const val ENDOFTEXT = "<|endoftext|>"
    }

    init {
        tokens = tokenize(rawText)
        vocab = buildTokenToId(tokens)
        idToToken = vocab.entries.associate { (token, id) -> id to token }
    }

    fun encode(text: String): List<Int> {
        val unkId = vocab[UNKNOWN] ?: error("UNK-Token fehlt")
        return tokenize(text).map { token -> vocab[token] ?: unkId }
    }

    fun decode(ids: List<Int>): String {
        val tokens = ids.map { id -> idToToken[id] ?: error("Unbekannte Token-ID: $id") }
        return joinTokens(tokens)
    }

    private fun joinTokens(tokens: List<String>): String {
        if (tokens.isEmpty()) return ""

        val noSpaceBefore = setOf(".", ",", "!", "?", ";", ":", ")", "]", "}", "\"", ENDOFTEXT, UNKNOWN)
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
                    token in noSpaceBefore -> false
                    prevToken in noSpaceAfter -> false
                    prevToken == "\"" && quoteIsOpen -> false
                    else -> true
                }

            if (needsSpace) out.append(' ')
            out.append(token)

            if (token == "\"") quoteIsOpen = !quoteIsOpen
        }

        return out.toString()
    }

    private fun tokenAtEnd(out: StringBuilder): String = out.last().toString()

    private fun tokenize(rawText: String): List<String> = tokenRegex.findAll(rawText).map { it.value }.toList()

    private fun buildTokenToId(tokens: List<String>): Map<String, Int> {
        val tokenToId = LinkedHashMap<String, Int>()
        for (token in tokens) {
            if (token !in tokenToId) {
                tokenToId[token] = tokenToId.size
            }
        }
        tokenToId[UNKNOWN] = tokenToId.size
        tokenToId[ENDOFTEXT] = tokenToId.size
        return tokenToId
    }
}

