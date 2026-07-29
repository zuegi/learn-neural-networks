# learn-neural-networks

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin/Maven Multi-Module-Projekt zum Lernen von neuronalen Netzen und LLM-Grundlagen. Das Projekt enthält ein MNIST-Trainingsmodul und ein LLM-Modul mit einfachem Tokenizer.

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer

Build und Tests:

```bash
mvn clean test
```

MNIST-Training starten:

```bash
mvn -pl chapter-1 exec:java
```

Tokenizer-Demo ausführen. Die Demo endet aktuell beim eingebauten Unknown-Token-Beispiel mit `Token nicht im Vokabular: 'Hello'`.

```bash
mvn -pl llm exec:java -Dexec.mainClass=ch.zuegi.ml.llm.MainKtKt
```

## Module

| Modul | Beschreibung | README |
|-------|-------------|--------|
| `chapter-1` | MNIST-Trainingsanwendung mit einfachem neuronalen Netz in Kotlin | [README](chapter-1/README.md) |
| `llm` | Kotlin-Modul mit einfachem Regex-Tokenizer für LLM-Grundlagen | [README](llm/README.md) |

## Top 10 Libraries

| # | Library | Version |
|---|---------|---------|
| 1 | JUnit Jupiter | 5.10.0 |
| 2 | Kotlin Standard Library | 2.4.0 |
| 3 | Kotlin Test JUnit 5 | 2.4.0 |
| 4 | AssertJ Core | 3.26.3 |
| 5 | Kotlin Deep Learning Dataset | 0.5.2 |
| 6 | Kotlin Deep Learning ONNX | 0.5.2 |
| 7 | Kotlin Deep Learning TensorFlow | 0.5.2 |
| 8 | Kotlinx Coroutines Core | 1.11.0 |

## Projektstruktur

```text
.
├── pom.xml
├── chapter-1
│   ├── pom.xml
│   └── src/main
│       ├── kotlin
│       └── resources/mnist
└── llm
    ├── pom.xml
    └── src
        ├── main/kotlin/ch/zuegi/ml/llm
        ├── main/resources/text
        └── test/kotlin/ch/zuegi/ml/llm
```
