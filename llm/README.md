# learn-neural-networks-llm

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin-Modul für LLM-Grundlagen. Aktuell enthält es einen einfachen Regex-Tokenizer mit Vokabularaufbau, Encoding, Decoding sowie einen DataLoader, der Trainingsdaten für Next-Token-Prediction erzeugt.

## Beschreibung

Das Modul liest den Text `the-verdict.txt` aus den Ressourcen, baut daraus ein Vokabular und wandelt Eingabetexte in Token-IDs um. Unbekannte Tokens werden auf `<|unk|>` gemappt. Beim Decoding rekonstruiert es Text mit Satzzeichen- und Anführungszeichen-Spacing. Aus den Token-IDs erzeugt der `TextDataLoader` per Sliding-Window Input/Target-Paare.

- `Main.kt`: Demo-Einstiegspunkt, lädt Textressource, tokenisiert und baut Trainingsdaten
- `SimpleTokenizerV1.kt`: Tokenisierung, Vokabular, Encoding, Decoding
- `TextDataLoader.kt`: Sliding-Window über Token-IDs, erzeugt `TrainingSample`s und Batches
- `SimpleTokenizerV1Test.kt`: Tests für bekannte Tokens, unbekannte Tokens und Roundtrip
- `src/main/resources/text/the-verdict.txt`: Trainings-/Vokabulartext

### Datenfluss

```text
Text
  → encode()        → [12, 45, 8, 91, ...]
  → TextDataLoader  → input/target Sequenzen
  → batches()       → [batchSize, contextLength]
```

### Nächster Schritt

Embeddings: Token-IDs → lernbare Vektoren (`[vocabSize, embeddingDim]`), danach Positional-Embeddings für Reihenfolge-Information.

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
│   └── TextDataLoader.kt
├── main/resources/text
│   └── the-verdict.txt
└── test/kotlin/ch/zuegi/ml/llm
    └── SimpleTokenizerV1Test.kt
```

## Konfiguration

Keine `application.yml` vorhanden. Der Ressourcenpfad ist aktuell in `Main.kt` als `/text/the-verdict.txt` festgelegt.
