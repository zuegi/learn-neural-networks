


Epoch 29 : 9510 / 10000
Dauer: 1m 28.931266709s

# Mit opimierter Version
- Hot-Path in matVecMul/transposeMatVecMul per Loops optimieren
- Lambda-Overhead (DoubleArray {}) in Hot-Path vermeiden
- Optional In-Place-Varianten ergänzen für späteren weiteren Speedup
- API-kompatibel zu deinem aktuellen Network.kt

Epoch 29 : 9521 / 10000
Dauer: 1m 24.952630041s



Die gezielte feedforward/backprop-Verbesserung mit weniger Allokationen, ohne Lernlogik zu ändern.
- NumKo.kt: cache-freundliche Kernops + in-place Ops
- Network.kt: backprop ohne outerProduct-Allokationen
- feedforward: kleine Vereinfachung, gleiche Semantik
- Ergebnis: weniger GC, schneller pro Batch

Epoch 27 : 9528 / 10000
Epoch 28 : 9524 / 10000
Epoch 29 : 9521 / 10000
Dauer: 1m 25.755988208s


batchsize 128
Epoch 29 : 9444 / 10000
Dauer: 1m 25.638707875s


# sgdParallel-Pfad, ohne Architekturumbau.
- Overhead in updateMiniBatchParallel senken
- Allokationen im Gewichtsupdate vermeiden (in-place statt Neuaufbau)
- Coroutine-Last begrenzen (Chunking)
- backprop im schnellen, direkten Stil halten

- Epoch 29 : 9467 / 10000
Dauer: 1m 7.799857959s

# Stark — hier ist präziser Patch für deine nur-parallele Network.kt, genau auf Performance getrimmt.
- updateMiniBatchParallel auf Chunk-Worker umbauen (weniger Coroutine-Overhead)
- In-place Update für weights/biases (keine Re-Allokation pro Batch)
- backprop unnötige addInPlace entfernen (direkte Zuweisung)

Epoch 29 : 9441 / 10000
Dauer: 1m 7.892177458s



# Top — ich gebe dir konkrete minimale Code-Änderung für limitedParallelism im Parallelpfad.
- sgdParallel um workers erweitern
- updateMiniBatchParallel mit begrenztem Dispatcher
- main-Aufruf anpassen
- mit 6 cores
Epoch 29 : 9450 / 10000
Dauer: 1m 8.340957792s

- mit 8 cores 
Epoch 29 : 9457 / 10000
Dauer: 1m 3.447483667s


# 8 Cores und nur 1 hidden layer
val sizes: IntArray = intArrayOf(784, 30, 10)
val epochs = 30
val miniBatchSize = 96
val workers = 8 // cores
val learningRate = 3.0

Epoch 29 : 9372 / 10000
Dauer: 18.739517125s

# learningRate 1.0
val sizes: IntArray = intArrayOf(784, 30, 10)
val epochs = 30
val miniBatchSize = 96
val workers = 8 // cores
val learningRate = 1.0

Epoch 29 : 9208 / 10000
Dauer: 18.760419125s

# learningRate 4.0
val sizes: IntArray = intArrayOf(784, 30, 10)
val epochs = 30
val miniBatchSize = 96
val workers = 8 // cores
val learningRate = 4.0

Epoch 29 : 9416 / 10000
Dauer: 18.813975500s

# 2 hidden layers learningRate 4.0
val sizes: IntArray = intArrayOf(784, 100, 30, 10)
val epochs = 30
val miniBatchSize = 64
val workers = 8 // cores
val learningRate = 4.0

Epoch 29 : 9525 / 10000
Dauer: 1m 6.857773583s

# learing rate 3.0
val sizes: IntArray = intArrayOf(784, 100, 30, 10)
val epochs = 30
val miniBatchSize = 64
val workers = 8 // cores
val learningRate = 5.0

learingRate (eta) ist der Gradient Descent: eta bestimmt, wie groß der Schritt in Richtung Minimum ist.
eta klein → kleine Schritte, langsames aber stabiles Lernen
eta groß → große Schritte, schneller aber kann überschiessen → schlechtere Accuracy

Epoch 29 : 9539 / 10000
Dauer: 1m 7.748968792s


# 3 hidden layers und 6.0 eta
val sizes: IntArray = intArrayOf(784, 200, 100, 30, 10)
val epochs = 30
val miniBatchSize = 64
val workers = 8 // cores
val learningRate = 6.0

Epoch 29 : 9573 / 10000
Dauer: 2m 22.877063125s