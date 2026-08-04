# learn-neural-networks-llm

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin-Modul für LLM-Grundlagen. Es enthält einen einfachen Regex-Tokenizer mit Vokabularaufbau, einen DataLoader für Next-Token-Prediction, lernbare Token- und Positional-Embeddings sowie eine einfache Single-Head Self-Attention.

## Beschreibung

Das Modul liest den Text `the-verdict.txt` aus den Ressourcen, baut daraus ein Vokabular und wandelt Eingabetexte in Token-IDs um. Unbekannte Tokens werden auf `<|unk|>` gemappt. Beim Decoding rekonstruiert es Text mit Satzzeichen- und Anführungszeichen-Spacing. Aus den Token-IDs erzeugt der `TextDataLoader` per Sliding-Window Input/Target-Paare. Die IDs werden über Embeddings in lernbare Vektoren überführt und von der Self-Attention kontextabhängig verarbeitet.

- `Main.kt`: Demo-Einstiegspunkt, lädt Textressource, tokenisiert und baut Trainingsdaten
- `SimpleTokenizerV1.kt`: Tokenisierung, Vokabular, Encoding, Decoding
- `TextDataLoader.kt`: Sliding-Window über Token-IDs, erzeugt `TrainingSample`s und Batches
- `TokenEmbedding.kt`: lernbare Token-Embedding-Tabelle `[vocabSize, embeddingDim]`, Zeilen-Lookup pro Token-ID
- `PositionalEmbedding.kt`: lernbare Positional-Embedding-Tabelle `[contextLength, embeddingDim]`, Lookup pro Position
- `InputEmbedding.kt`: addiert Token- und Positional-Embeddings elementweise zu Input-Embeddings
- `SelfAttention.kt`: Single-Head Self-Attention (Forward-Pass) über Input-Embeddings
- `SimpleTokenizerV1Test.kt`, `TokenEmbeddingTest.kt`: Tests
- `src/main/resources/text/the-verdict.txt`: Trainings-/Vokabulartext

### Datenfluss

```text
Text
  → encode()          → [12, 45, 8, 91, ...]
  → TextDataLoader    → input/target Sequenzen
  → TokenEmbedding    → [contextLength, embeddingDim]
  + PositionalEmbedding → [contextLength, embeddingDim]
  = InputEmbedding    → [contextLength, embeddingDim]
  → SelfAttention     → [contextLength, dK]
```

### Warum `DoubleArray`

Der numerische Kern nutzt durchgehend `DoubleArray` statt `List<Double>` oder `Array<Double>`:

- `DoubleArray` ist ein primitives `double[]` — kompakt und zusammenhängend im Speicher (cache-freundlich)
- `Array<Double>` und `List<Double>` boxen jede Zahl zu einem `java.lang.Double`-Objekt: mehr Speicher, Pointer-Dereferenzierung, mehr GC-Druck
- in engen Schleifen wie `sum += a[i][k] * b[k][j]` entfällt so das Unboxing pro Multiplikation, und der JIT kann besser optimieren

Deshalb verwenden `TokenEmbedding`, `PositionalEmbedding`, `InputEmbedding` und `SelfAttention` denselben Typ wie das bestehende `Network.kt`. `List<Int>` bleibt dort, wo Performance zweitrangig ist (z.B. `tokenIds`).

### Nächster Schritt

Causal Masking für die Self-Attention, damit Tokens nicht auf zukünftige Positionen schauen. Danach Multi-Head-Attention, Transformer-Block und Training-Loop (Next-Token-Loss).

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer

Tests ausführen:

```bash
mvn -pl llm test
```

Demo ausführen. Sie tokenisiert den Text und gibt Größe sowie erstes Input/Target-Sample des `TextDataLoader` aus.

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
│   └── SelfAttention.kt
├── main/resources/text
│   └── the-verdict.txt
└── test/kotlin/ch/zuegi/ml/llm
    ├── SimpleTokenizerV1Test.kt
    └── TokenEmbeddingTest.kt
```

## Konfiguration

Keine `application.yml` vorhanden. Der Ressourcenpfad ist aktuell in `Main.kt` als `/text/the-verdict.txt` festgelegt.
