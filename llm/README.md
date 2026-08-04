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
- `SimpleTokenizerV1Test.kt`, `TokenEmbeddingTest.kt`, `SelfAttentionTest.kt`, `MultiHeadAttentionTest.kt`, `LayerNormTest.kt`, `FeedForwardTest.kt`, `TransformerBlockTest.kt`, `GPTModelTest.kt`: Tests
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

### Nächster Schritt

Der komplette Forward-Pfad steht: `GPTModel` bildet Token-IDs auf Logits ab und erzeugt über `generate()` autoregressiv neue Tokens (Greedy oder Sampling mit Temperature). Da das Modell noch untrainiert ist, ist die Ausgabe zufällig — der Mechanismus ist aber vollständig. Als Nächstes folgt der Trainingsteil: Cross-Entropy-Loss über die Logits und Backpropagation, geplant als kleines Autograd-System mit numerischem Gradient-Check.

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
│   └── GPTModel.kt
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
    └── GPTModelTest.kt
```

## Konfiguration

Keine `application.yml` vorhanden. Der Ressourcenpfad ist aktuell in `Main.kt` als `/text/the-verdict.txt` festgelegt.
