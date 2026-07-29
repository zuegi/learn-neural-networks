# learn-neural-networks

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin/Maven Multi-Module-Projekt zum Lernen neuronaler Netze. Aktuell enthält es eine MNIST-Trainingsanwendung mit eigener Feedforward-, Backpropagation- und SGD-Implementierung.

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer

Build:

```bash
mvn clean test
```

Anwendung starten:

```bash
mvn -pl chapter-1 exec:java
```

## Module

| Modul | Beschreibung | README |
|-------|-------------|--------|
| `chapter-1` | MNIST-Trainingsanwendung mit einfachem neuronalen Netz in Kotlin | [README](chapter-1/README.md) |
| `llm` | Kotlin-Modul für spätere LLM-Experimente | [README](llm/README.md) |

## Top 10 Libraries

| # | Library | Version |
|---|---------|---------|
| 1 | JUnit Jupiter | 5.10.0 |
| 2 | Kotlin Standard Library | 2.4.0 |
| 3 | Kotlin Test JUnit 5 | 2.4.0 |
| 4 | Kotlin Deep Learning Dataset | 0.5.2 |
| 5 | Kotlin Deep Learning ONNX | 0.5.2 |
| 6 | Kotlin Deep Learning TensorFlow | 0.5.2 |
| 7 | Kotlinx Coroutines Core | 1.11.0 |

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
        ├── main/kotlin
        └── test/kotlin
```
