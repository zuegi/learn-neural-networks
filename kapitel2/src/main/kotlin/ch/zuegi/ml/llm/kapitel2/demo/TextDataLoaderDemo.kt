package ch.zuegi.ml.llm.kapitel2.demo

import ch.zuegi.ml.llm.shared.TextDataLoader

fun main() {
    proveSamplesAreBuiltCorrectly()
    println()
    proveBatchesAreGroupedCorrectly()
    println()
    proveSizeCountsWindowsCorrectly()
}

// tag::proveSamples[]
private fun proveSamplesAreBuiltCorrectly() {
    println("*** samples are built correctly ***")
    val tokenIds = listOf(10, 11, 12, 13, 14, 15)
    val loader = TextDataLoader(tokenIds, contextLength = 4)

    val samples = loader.samples()

    println("tokenIds: $tokenIds")
    println("samples: $samples")

    check(samples.size == 2) { "Unerwartete Anzahl Samples: ${samples.size}" }
    check(samples[0].inputIds == listOf(10, 11, 12, 13)) { "Unerwartete inputIds im ersten Sample: ${samples[0].inputIds}" }
    check(samples[0].targetIds == listOf(11, 12, 13, 14)) { "Unerwartete targetIds im ersten Sample: ${samples[0].targetIds}" }
    check(samples[1].inputIds == listOf(11, 12, 13, 14)) { "Unerwartete inputIds im zweiten Sample: ${samples[1].inputIds}" }
    check(samples[1].targetIds == listOf(12, 13, 14, 15)) { "Unerwartete targetIds im zweiten Sample: ${samples[1].targetIds}" }
}
// end::proveSamples[]

// tag::proveBatches[]
private fun proveBatchesAreGroupedCorrectly() {
    println("*** batches are grouped correctly ***")
    val tokenIds = listOf(0, 1, 2, 3, 4, 5)
    val loader = TextDataLoader(tokenIds, contextLength = 2, batchSize = 2)

    val batches = loader.batches()

    println("tokenIds: $tokenIds")
    println("batches: $batches")

    check(batches.size == 2) { "Unerwartete Anzahl Batches: ${batches.size}" }
    check(batches[0].size == 2) { "Unerwartete Größe des ersten Batches: ${batches[0].size}" }
    check(batches[1].size == 2) { "Unerwartete Größe des zweiten Batches: ${batches[1].size}" }
}
// end::proveBatches[]

// tag::proveSize[]
private fun proveSizeCountsWindowsCorrectly() {
    println("*** size counts windows correctly ***")
    val tokenIds = listOf(0, 1, 2, 3, 4, 5, 6)
    val loader = TextDataLoader(tokenIds, contextLength = 3, stride = 2)

    println("tokenIds: $tokenIds")
    println("size: ${loader.size()}")

    check(loader.size() == 2) { "Unerwartete Fensteranzahl: ${loader.size()}" }
}
// end::proveSize[]

