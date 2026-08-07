package ch.zuegi.ml.llm.kapitel3.scratch

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
 * Beispiel:
 *
 * val tokenizer = SimpleTokenizerV1(""It's the last painted," you know.")
 *
 * val ids = tokenizer.encode(""It's unknown."")
 * val text = tokenizer.decode(ids)
 *
 *
 * @param rawText Text, aus dem das Vokabular aufgebaut wird.
 */

class SimpleTokenizerV1(
    rawText: String,
) {
    private val tokenRegex = Regex("""<\|unk\|>|<\|endoftext\|>|\p{L}+(?:[_'’\-]\p{L}+)*|[.,!?;:"()]""")
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

    /**
     * Wandelt Text in Token-IDs um.
     *
     * Der Text wird mit derselben Regex tokenisiert, die auch für den Aufbau des Vokabulars
     * verwendet wurde. Bekannte Tokens werden über das Vokabular in ihre IDs übersetzt.
     * Unbekannte Tokens werden auf die ID von [UNKNOWN] gemappt.
     *
     * [ENDOFTEXT] wird nur dann als eigene ID ausgegeben, wenn der Eingabetext
     * das Token `<|endoftext|>` explizit enthält.
     *
     * @param text Text, der encodiert werden soll.
     * @return Liste von Token-IDs.
     */
    fun encode(text: String): List<Int> {
        val unkId = vocab[UNKNOWN] ?: error("UNK-Token fehlt")
        return tokenize(text).map { token -> vocab[token] ?: unkId }
    }

    /**
     * Wandelt Token-IDs zurück in Text.
     *
     * Jede ID wird über die interne Reverse-Map in ihr Token zurückübersetzt.
     * Danach werden die Tokens mit einfachen Spacing-Regeln zusammengesetzt:
     * - kein Leerzeichen vor Satzzeichen wie `.`, `,`, `!`, `?`
     * - kein Leerzeichen vor [UNKNOWN] und [ENDOFTEXT]
     * - einfache Behandlung von Anführungszeichen
     *
     * @param ids Token-IDs, die decodiert werden sollen.
     * @return rekonstruierter Text.
     * @throws IllegalStateException wenn eine ID nicht im Vokabular existiert.
     */
    fun decode(ids: List<Int>): String {
        val tokens =
            ids.map { id ->
                idToToken[id] ?: error("Unbekannte Token-ID: $id")
            }
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

    /**
     * Zerlegt Text in Tokens.
     *
     * Die Regex erkennt:
     * - Special Tokens: `<|unk|>`, `<|endoftext|>`
     * - Wörter mit Unicode-Buchstaben
     * - interne Zeichen wie `_`, `'`, `’`, `-`
     * - einzelne Satzzeichen
     *
     * Leerzeichen werden nicht als Tokens übernommen.
     *
     * @param rawText Text, der tokenisiert werden soll.
     * @return Liste erkannter Tokens in Originalreihenfolge.
     */
    private fun tokenize(rawText: String): List<String> = tokenRegex.findAll(rawText).map { it.value }.toList()

    /**
     * Erstellt das Vokabular `Token -> ID`.
     *
     * Tokens erhalten IDs in der Reihenfolge ihres ersten Auftretens.
     * Danach werden [UNKNOWN] und [ENDOFTEXT] immer am Ende ergänzt.
     *
     * @param tokens Tokens aus dem Trainings-/Rohtext.
     * @return stabile Map von Token zu numerischer ID.
     */
    private fun buildTokenToId(tokens: List<String>): Map<String, Int> {
        val tokenToId = LinkedHashMap<String, Int>() // Reihenfolge stabil
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
