# learn-neural-networks-llm

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin-Modul für LLM-Grundlagen. Es enthält einen einfachen Regex-Tokenizer mit Vokabularaufbau, einen DataLoader für Next-Token-Prediction, lernbare Token- und Positional-Embeddings sowie die Transformer-Bausteine Self-Attention (mit optionalem Causal Masking), Multi-Head-Attention, Layer Normalization und ein Feed-Forward-Netz.

## Beschreibung

Das Modul liest den Text `the-verdict.txt` aus den Ressourcen, baut daraus ein Vokabular und wandelt Eingabetexte in Token-IDs um. Unbekannte Tokens werden auf `<|unk|>` gemappt. Beim Decoding rekonstruiert es Text mit Satzzeichen- und Anführungszeichen-Spacing. Aus den Token-IDs erzeugt der `TextDataLoader` per Sliding-Window Input/Target-Paare. Die IDs werden über Embeddings in lernbare Vektoren überführt und von den Attention- und Feed-Forward-Bausteinen kontextabhängig verarbeitet. Alle Bausteine implementieren aktuell nur den Forward-Pass.

- `Main.kt`: Demo-Einstiegspunkt, lädt Textressource, tokenisiert und baut Trainingsdaten
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
- `autograd/Value.kt`: skalarer Autograd-Knoten (Micrograd-Stil) mit `+`, `*`, `/`, `pow`, `tanh`, Skalar-Operatoren und `backward()`
- `autograd/Tensor.kt`: Autograd-Knoten für 1D-Vektoren mit elementweisen Ops, `matVecMul`, `matMul` und `softmaxCrossEntropy`
- `SimpleTokenizerV1Test.kt`, `TokenEmbeddingTest.kt`, `SelfAttentionTest.kt`, `MultiHeadAttentionTest.kt`, `LayerNormTest.kt`, `FeedForwardTest.kt`, `TransformerBlockTest.kt`, `GPTModelTest.kt`: Tests der Modellbausteine
- `autograd/ValueTest.kt`, `autograd/TensorTest.kt`, `autograd/TensorMathVecTest.kt`, `autograd/TensorMatMulTest.kt`, `autograd/TensorSoftmaxCrossEntropyTest.kt`: Autograd-Tests mit numerischem Gradient-Check
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

- elementweise Ops (`+`, `*`, `tanh`)
- `matVecMul` (Matrix × Vektor): Backward `dx = Wᵀ·dy`, `dW = dy⊗x`
- `matMul` (Matrix × Matrix): Backward `dA = dC·Bᵀ`, `dB = Aᵀ·dC`
- `softmaxCrossEntropy`: Softmax und Cross-Entropy als eine Operation, mit dem eleganten Gradienten `dLogits = softmax(logits) − oneHot(target)` (umgeht die volle Softmax-Jacobian)

Matrizen werden flach (row-major) in `DoubleArray` gehalten; die Form wird über explizite Dimensionsparameter übergeben.

**Gradient-Check als Sicherheitsnetz:** Jede Operation wird gegen die numerische Ableitung `(f(x+h) − f(x−h)) / 2h` geprüft. Stimmt der analytische Gradient mit dem numerischen überein, ist die Backward-Regel bewiesen — kein blindes Debuggen.

### Nächster Schritt

Das Autograd-Fundament steht: `Value` (Skalar) und `Tensor` (Vektor/Matrix) mit `matVecMul`, `matMul` und `softmaxCrossEntropy`, alle per Gradient-Check verifiziert. Als Nächstes folgen ein einfacher Optimizer (SGD: `param.data -= learningRate * param.grad`) und ein Training-Loop (forward → loss → `backward()` → update). Danach der größere Umbau: die GPT-Bausteine (`LayerNorm`, `FeedForward`, Attention) von rohem `DoubleArray` auf `Tensor` umstellen, damit das gesamte `GPTModel` trainierbar wird.

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer

Tests ausführen:

```bash
mvn -pl llm test
```

Demo ausführen. Sie tokenisiert den Text, baut ein `GPTModel` und erzeugt aus einer Start-Sequenz autoregressiv neue Tokens (greedy). Da das Modell untrainiert ist, ist der erzeugte Text zufällig.

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
│       └── Tensor.kt
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
        └── TensorSoftmaxCrossEntropyTest.kt
```

## Konfiguration

Keine `application.yml` vorhanden. Der Ressourcenpfad ist aktuell in `Main.kt` als `/text/the-verdict.txt` festgelegt.
