package ch.zuegi.ml.llm.shared

fun text2TokenIds(
    text: String,
    tokenizer: Tokenizer,
): List<Int> = tokenizer.encode(text)

fun tokenIds2Text(
    ids: List<Int>,
    tokenizer: Tokenizer,
): String = tokenizer.decode(ids)

/**
 * Raschka-style greedy generation (simple):
 * - no temperature
 * - no top-k/top-p
 * - no eos stop
 */
fun generateTextSimple(
    model: (List<Int>) -> Array<FloatArray>, // logits: [seqLen][vocabSize]
    startTokenIds: List<Int>,
    maxNewTokens: Int,
    contextSize: Int,
): List<Int> {
    val generated = startTokenIds.toMutableList()

    repeat(maxNewTokens) {
        // 1) Context window (like idx[:, -context_size:])
        val idxCond =
            if (generated.size > contextSize) {
                generated.takeLast(contextSize)
            } else {
                generated.toList()
            }

        // 2) Forward pass: logits for each position
        val logitsPerPosition: Array<FloatArray> = model(idxCond)

        // 3) Take last position logits (next-token distribution)
        val lastLogits: FloatArray = logitsPerPosition.last()

        // 4) Greedy pick (argmax)
        val nextTokenId: Int = argmax(lastLogits)

        // 5) Append
        generated += nextTokenId
    }

    return generated
}

fun argmax(values: FloatArray): Int {
    var bestIndex = 0
    var bestValue = values[0]
    for (i in 1 until values.size) {
        if (values[i] > bestValue) {
            bestValue = values[i]
            bestIndex = i
        }
    }
    return bestIndex
}
