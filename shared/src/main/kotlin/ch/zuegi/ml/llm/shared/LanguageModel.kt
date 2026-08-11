package ch.zuegi.ml.llm.shared

import kotlin.random.Random

interface LanguageModel {
    /**
     * Input: token ids of current context.
     * Output: logits per position: [seqLen][vocabSize]
     */
    fun forward(tokenIds: List<Int>): Array<FloatArray>
}

class DummyLanguageModel(
    private val vocabSize: Int,
    private val seed: Int = 42,
) : LanguageModel {
    private val rng = Random(seed)

    override fun forward(tokenIds: List<Int>): Array<FloatArray> =
        Array(tokenIds.size) {
            FloatArray(vocabSize) { rng.nextFloat() }
        }
}

fun generateTextSimple(
    model: LanguageModel,
    startTokenIds: List<Int>,
    maxNewTokens: Int,
    contextSize: Int,
): List<Int> {
    val generated = startTokenIds.toMutableList()

    repeat(maxNewTokens) {
        val idxCond =
            if (generated.size > contextSize) {
                generated.takeLast(contextSize)
            } else {
                generated.toList()
            }

        val logitsPerPosition = model.forward(idxCond)
        val lastLogits = logitsPerPosition.last()
        val nextTokenId = argmaxIndex(lastLogits)
        generated += nextTokenId
    }

    return generated
}

fun argmaxIndex(values: FloatArray): Int {
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
