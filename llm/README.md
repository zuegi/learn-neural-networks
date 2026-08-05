# learn-neural-networks-llm

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin-Modul für LLM-Grundlagen. Es enthält einen einfachen Regex-Tokenizer mit Vokabularaufbau, einen DataLoader für Next-Token-Prediction, lernbare Token- und Positional-Embeddings, die Transformer-Bausteine Self-Attention (mit optionalem Causal Masking), Multi-Head-Attention, Layer Normalization und ein Feed-Forward-Netz sowie ein eigenes Autograd-System, mit dem ein komplettes GPT von Grund auf trainiert wird.

## Beschreibung

Das Modul liest den Text `the-verdict.txt` aus den Ressourcen, baut daraus ein Vokabular und wandelt Eingabetexte in Token-IDs um. Unbekannte Tokens werden auf `<|unk|>` gemappt. Beim Decoding rekonstruiert es Text mit Satzzeichen- und Anführungszeichen-Spacing. Aus den Token-IDs erzeugt der `TextDataLoader` per Sliding-Window Input/Target-Paare. Die IDs werden über Embeddings in lernbare Vektoren überführt und von den Attention- und Feed-Forward-Bausteinen kontextabhängig verarbeitet.

Es gibt zwei parallele Ausbaustufen der Modellbausteine:

- **Forward-only** (`SelfAttention`, `MultiHeadAttention`, `LayerNorm`, `FeedForward`, `TransformerBlock`, `GPTModel`): arbeiten direkt auf `Array<DoubleArray>`, dienen als lesbare Referenz und Vergleichs-Orakel.
- **Trainierbar** (`autograd/*Layer`): bauen auf dem `Tensor`-Autograd auf, halten lernbare Parameter und liefern über `parameters()` die Gewichte für den Optimizer. Damit ist das gesamte `GPTModelLayer` end-to-end trainierbar.

Forward-only Bausteine:

- `Main.kt`: Einstiegspunkt. `main()` zeigt den untrainierten Forward-only-`GPTModel`-Pfad, `mainTrain()` trainiert `GPTModelLayer` auf `the-verdict.txt` und generiert Text
- `SimpleTokenizerV1.kt`: Tokenisierung, Vokabular, Encoding, Decoding
- `TextDataLoader.kt`: Sliding-Window über Token-IDs, erzeugt `TrainingSample`s und Batches
- `TokenEmbedding.kt`: lernbare Token-Embedding-Tabelle `[vocabSize, embeddingDim]`, Zeilen-Lookup pro Token-ID
- `PositionalEmbedding.kt`: lernbare Positional-Embedding-Tabelle `[contextLength, embeddingDim]`, Lookup pro Position
- `InputEmbedding.kt`: addiert Token- und Positional-Embeddings elementweise zu Input-Embeddings
- `SelfAttention.kt`: Single-Head Self-Attention (Forward-Pass), optional mit Causal Masking (`j <= i`)
- `MultiHeadAttention.kt`: mehrere parallele Attention-Köpfe, Konkatenation und Output-Projektion `Wo` zurück auf `embeddingDim`
- `LayerNorm.kt`: Layer Normalization pro Token-Zeile mit lernbaren Parametern `gamma`/`beta`
- `FeedForward.kt`: position-weises Feed-Forward-Netz mit zwei linearen Schichten und GELU-Aktivierung
- `TransformerBlock.kt`: kombiniert Multi-Head-Attention und Feed-Forward mit Pre-LN und Residual-Verbindungen
- `GPTModel.kt`: End-to-End-Forward von Token-IDs zu Logits (Embeddings → N Transformer-Blöcke → finale LayerNorm → Output-Projektion) sowie `generate()` für autoregressive Token-Erzeugung (Greedy oder Sampling mit Temperature)

Autograd und trainierbare Bausteine (Paket `autograd`):

- `Value.kt`: skalarer Autograd-Knoten (Micrograd-Stil) mit `+`, `*`, `/`, `pow`, `tanh`, Skalar-Operatoren und `backward()`
- `Tensor.kt`: Autograd-Knoten für 1D-Vektoren/flache Matrizen mit elementweisen Ops (`+`, `*`, `tanh`, `gelu`), `matVecMul`, `matMul`, `softmax`, `softmaxCrossEntropy`, `layerNorm` sowie den Struktur-Ops `row`, `transposeMatrix`, `maskCausalScale`, `scale`, `embeddingLookup`, `stackRows`, `concatCols`
- `SGD.kt`: Optimizer (`param.data -= learningRate * param.grad`) mit `zeroGrad()`
- `LayerNormLayer.kt`: trainierbare LayerNorm pro Token-Zeile (`gamma`/`beta` als `Tensor`)
- `FeedForwardLayer.kt`: trainierbares Feed-Forward-Netz (zwei lineare Schichten + GELU) auf `Tensor`
- `SelfAttentionLayer.kt`: trainierbare Single-Head Self-Attention, optional Causal
- `MultiHeadAttentionLayer.kt`: mehrere `SelfAttentionLayer` + Output-Projektion `Wo`
- `TransformerBlockLayer.kt`: Pre-LN + Residual über `MultiHeadAttentionLayer` und `FeedForwardLayer`
- `GPTModelLayer.kt`: trainierbares GPT (Embeddings → N Blöcke → finale LN → Output-Projektion) mit `loss()` (mittlerer Cross-Entropy) und `generate()` (greedy oder Sampling mit Temperature/top-k)

Tests:

- `SimpleTokenizerV1Test.kt`, `TokenEmbeddingTest.kt`, `SelfAttentionTest.kt`, `MultiHeadAttentionTest.kt`, `LayerNormTest.kt`, `FeedForwardTest.kt`, `TransformerBlockTest.kt`, `GPTModelTest.kt`: Tests der Forward-only-Bausteine
- `autograd/ValueTest.kt`, `TensorTest.kt`, `TensorMathVecTest.kt`, `TensorMatMulTest.kt`, `TensorSoftmaxCrossEntropyTest.kt`, `TensorSoftmaxTest.kt`, `TensorLayerNormTest.kt`, `TensorGeluTest.kt`, `SGDTest.kt`: Autograd-Ops mit numerischem Gradient-Check
- `autograd/LayerNormLayerTest.kt`, `FeedForwardLayerTest.kt`, `SelfAttentionLayerTest.kt`, `MultiHeadAttentionLayerTest.kt`, `TransformerBlockLayerTest.kt`, `GPTModelLayerTest.kt`: trainierbare Layer (Form-Checks und "Loss fällt beim Training")
- `src/main/resources/text/the-verdict.txt`: Trainings-/Vokabulartext

### Datenfluss

```text
Text
  → encode()          → [12, 45, 8, 91, ...]
  → TextDataLoader    → input/target Sequenzen
  → TokenEmbedding    → [contextLength, embeddingDim]
  + PositionalEmbedding → [contextLength, embeddingDim]
  = InputEmbedding    → [contextLength, embeddingDim]
  → N × TransformerBlock → [contextLength, embeddingDim]   (Pre-LN: MHA + FeedForward, je Residual)
  → finale LayerNorm  → [contextLength, embeddingDim]
  → Output-Projektion → [contextLength, vocabSize]         (Logits, via GPTModel)
```

Alle Bausteine ab `InputEmbedding` bis zur finalen Norm behalten die Form `[contextLength, embeddingDim]`, sodass Residual-Verbindungen greifen und sich mehrere Blöcke stapeln lassen. Erst die Output-Projektion in `GPTModel` bildet auf `vocabSize` ab.

Im trainierbaren Pfad (`GPTModelLayer`) ist der Ablauf identisch, nur werden alle Zwischengrößen als `Tensor` (flache row-major Matrix) durch den Autograd-Graphen geführt. Aus den Logits berechnet `loss()` den mittleren `softmaxCrossEntropy` gegen die Ziel-Tokens; `backward()` propagiert die Gradienten bis in die Embedding-Tabellen zurück, `SGD.step()` aktualisiert alle Parameter.

### Warum `DoubleArray`

Der numerische Kern nutzt durchgehend `DoubleArray` statt `List<Double>` oder `Array<Double>`:

- `DoubleArray` ist ein primitives `double[]` — kompakt und zusammenhängend im Speicher (cache-freundlich)
- `Array<Double>` und `List<Double>` boxen jede Zahl zu einem `java.lang.Double`-Objekt: mehr Speicher, Pointer-Dereferenzierung, mehr GC-Druck
- in engen Schleifen wie `sum += a[i][k] * b[k][j]` entfällt so das Unboxing pro Multiplikation, und der JIT kann besser optimieren

Deshalb verwenden die Embeddings, `SelfAttention`, `MultiHeadAttention`, `LayerNorm` und `FeedForward` denselben Typ wie das bestehende `Network.kt`. `List<Int>` bleibt dort, wo Performance zweitrangig ist (z.B. `tokenIds`).

### Aktivierungsfunktion: GELU (tanh-Approximation)

`FeedForward` nutzt als Aktivierung **GELU** (Gaussian Error Linear Unit), nicht ReLU:

- GELU ist überall glatt/differenzierbar — wichtig für saubere Gradienten beim späteren Training
- GELU lässt kleine negative Werte durch, statt sie wie ReLU hart auf 0 zu schneiden (weniger "tote" Neuronen)
- GELU ist die Standard-Aktivierung in Transformer-Modellen (GPT, BERT)

Verwendet wird die **tanh-Approximation** (wie in GPT-2), nicht das exakte GELU:

```text
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
```

Grund: Das exakte GELU basiert auf der Gauß'schen Fehlerfunktion `erf`

```text
GELU(x) = 0.5 · x · (1 + erf(x / √2))
```

und `erf` ist in der Kotlin-/Java-Standardbibliothek nicht verfügbar. Die tanh-Approximation kommt mit `kotlin.math.tanh` aus, weicht nur um ~1e-3 vom exakten Wert ab und wird von GPT-2 selbst genutzt — für dieses Lernprojekt die pragmatisch beste Wahl.

### Autograd (automatisches Differenzieren)

Für das Training wird ein eigenes kleines Autograd-System gebaut — die Grundlage, um Gradienten automatisch zu berechnen (das, was PyTorchs `loss.backward()` intern tut). Jede Operation merkt sich ihre lokale Ableitung; `backward()` propagiert den Gradienten in topologischer Reihenfolge rückwärts durch den Rechengraphen. Gradienten werden akkumuliert (`+=`), damit geteilte Knoten die Beiträge aus allen Pfaden aufsummieren.

`autograd/Value.kt` — **skalares** Autograd (Micrograd-Stil) zum Verstehen des Prinzips:

- Operationen: `+`, `*`, `/`, `pow`, `tanh`
- Skalar-Varianten (`x * 3.0`, `2.0 * x`) über Member-Operatoren und Top-Level-Extensions (Kotlin-Äquivalent zu Pythons `__rmul__`/`__radd__`/`__rtruediv__`)

`autograd/Tensor.kt` — **Vektor/Matrix**-Autograd, schrittweise Richtung GPT-Training aufgebaut:

- elementweise Ops (`+`, `*`, `tanh`, `gelu`)
- `matVecMul` (Matrix × Vektor): Backward `dx = Wᵀ·dy`, `dW = dy⊗x`
- `matMul` (Matrix × Matrix): Backward `dA = dC·Bᵀ`, `dB = Aᵀ·dC`
- `softmax`: eigenständige Softmax mit voller Jacobian `dx_i = p_i·(dy_i − Σ_j p_j·dy_j)`
- `softmaxCrossEntropy`: Softmax und Cross-Entropy als eine Operation, mit dem eleganten Gradienten `dLogits = softmax(logits) − oneHot(target)` (umgeht die volle Softmax-Jacobian)
- `layerNorm`: Layer Normalization mit lernbaren `gamma`/`beta` und Standard-LayerNorm-Backward
- Struktur-Ops für Attention/Modell: `row` (Zeile aus flacher Matrix), `transposeMatrix`, `maskCausalScale` (Causal-Maske + Skalierung), `scale`, `embeddingLookup` (Tabellenzeile), `stackRows` (Zeilen → Matrix), `concatCols` (Matrizen spaltenweise konkatenieren)

Matrizen werden flach (row-major) in `DoubleArray` gehalten; die Form wird über explizite Dimensionsparameter übergeben.

**Gradient-Check als Sicherheitsnetz:** Jede Operation wird gegen die numerische Ableitung `(f(x+h) − f(x−h)) / 2h` geprüft. Stimmt der analytische Gradient mit dem numerischen überein, ist die Backward-Regel bewiesen — kein blindes Debuggen.

### Training

Mit `SGD` und den `*Layer`-Bausteinen ist das gesamte Modell trainierbar. Der Trainings-Loop (`mainTrain()` in `Main.kt`) folgt dem Standardmuster:

```text
für jede Epoche, für jedes TrainingSample:
    sgd.zeroGrad()                          // Gradienten zurücksetzen
    loss = model.loss(input, target)        // Forward + Cross-Entropy
    loss.backward()                         // Gradienten rückwärts propagieren
    sgd.step()                              // Parameter aktualisieren
```

Auf `the-verdict.txt` fällt der Loss so von ~6.96 (Zufalls-Niveau `ln(vocabSize)`) auf ~2.70 in 10 Epochen — der Nachweis, dass das selbstgebaute Autograd korrekt trainiert. Der generierte Text bleibt Kauderwelsch, weil Modell (embeddingDim 32, 2 Blöcke) und Datenmenge (wenige Samples) für ein Lernprojekt bewusst klein gehalten sind. `generate()` unterstützt greedy (deterministisch, neigt zu Wiederholungen) und Sampling mit `temperature`/`topK` (variabler, bricht Schleifen).

### Nächster Schritt

Das komplette trainierbare GPT steht: Tokenizer → `TextDataLoader` → `GPTModelLayer.loss` → `backward()` → `SGD` → `generate()`, alle Autograd-Ops per Gradient-Check verifiziert. Mögliche Erweiterungen: größeres Modell / mehr Trainingsdaten (ggf. Performance des Scalar/Tensor-Autograds optimieren), Batch-Verarbeitung mehrerer Samples, ein besserer Tokenizer (BPE) oder Lernraten-Scheduling.

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer

Tests ausführen:

```bash
mvn -pl llm test
```

Demo ausführen. `main()` tokenisiert den Text, baut ein untrainiertes Forward-only-`GPTModel` und erzeugt aus einer Start-Sequenz autoregressiv neue Tokens (der Text ist zufällig). `mainTrain()` trainiert stattdessen `GPTModelLayer` auf dem Text (Loss fällt pro Epoche) und generiert danach per Sampling.

```bash
mvn -pl llm exec:java -Dexec.mainClass=ch.zuegi.ml.llm.MainKt
```

## Architektur & Abhängigkeiten

Maven-Koordinaten:

```xml
<dependency>
    <groupId>ch.zuegi.machinelearning</groupId>
    <artifactId>learn-neural-networks-llm</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

Interne Modul-Abhängigkeiten:

| Modul | Zweck |
|-------|------|
| Keine | Standalone-Modul |

Externe Libraries:

| Library | Zweck |
|---------|------|
| Kotlin Standard Library | Kotlin-Laufzeit |
| Kotlin Test JUnit 5 | Kotlin-Testintegration |
| JUnit Jupiter | Test-Runner |
| AssertJ Core | Fluent Assertions in Tests |

Paketstruktur:

```text
src
├── main/kotlin/ch/zuegi/ml/llm
│   ├── Main.kt
│   ├── SimpleTokenizerV1.kt
│   ├── TrainingSample.kt
│   ├── TextDataLoader.kt
│   ├── TokenEmbedding.kt
│   ├── PositionalEmbedding.kt
│   ├── InputEmbedding.kt
│   ├── SelfAttention.kt
│   ├── MultiHeadAttention.kt
│   ├── LayerNorm.kt
│   ├── FeedForward.kt
│   ├── TransformerBlock.kt
│   ├── GPTModel.kt
│   └── autograd
│       ├── Value.kt
│       ├── Tensor.kt
│       ├── SGD.kt
│       ├── LayerNormLayer.kt
│       ├── FeedForwardLayer.kt
│       ├── SelfAttentionLayer.kt
│       ├── MultiHeadAttentionLayer.kt
│       ├── TransformerBlockLayer.kt
│       └── GPTModelLayer.kt
├── main/resources/text
│   └── the-verdict.txt
└── test/kotlin/ch/zuegi/ml/llm
    ├── SimpleTokenizerV1Test.kt
    ├── TokenEmbeddingTest.kt
    ├── SelfAttentionTest.kt
    ├── MultiHeadAttentionTest.kt
    ├── LayerNormTest.kt
    ├── FeedForwardTest.kt
    ├── TransformerBlockTest.kt
    ├── GPTModelTest.kt
    └── autograd
        ├── ValueTest.kt
        ├── TensorTest.kt
        ├── TensorMathVecTest.kt
        ├── TensorMatMulTest.kt
        ├── TensorSoftmaxCrossEntropyTest.kt
        ├── TensorSoftmaxTest.kt
        ├── TensorLayerNormTest.kt
        ├── TensorGeluTest.kt
        ├── SGDTest.kt
        ├── LayerNormLayerTest.kt
        ├── FeedForwardLayerTest.kt
        ├── SelfAttentionLayerTest.kt
        ├── MultiHeadAttentionLayerTest.kt
        ├── TransformerBlockLayerTest.kt
        └── GPTModelLayerTest.kt
```

## Konfiguration

Keine `application.yml` vorhanden. Der Ressourcenpfad ist aktuell in `Main.kt` als `/text/the-verdict.txt` festgelegt.
