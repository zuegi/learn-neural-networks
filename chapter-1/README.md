# learn-neural-networks-chapter-1

<!-- TODO: Beschreibung in pom.xml ergänzen -->

MNIST-Trainingsanwendung mit einem einfachen neuronalen Netz in Kotlin. Das Modul lädt IDX-Dateien aus den Ressourcen, trainiert ein Feedforward-Netz per Mini-Batch-SGD und gibt Accuracy pro Epoche aus.

## Beschreibung

Dieses Modul enthält den ausführbaren Einstiegspunkt für Kapitel 1 des Lernprojekts. Es implementiert Kernbausteine eines neuronalen Netzes ohne Framework-Abstraktion:

- `Network.kt`: Feedforward, Backpropagation, paralleles Mini-Batch-SGD, Evaluation
- `MnistLoader.kt`: Laden von MNIST IDX-Bildern und Labels aus `src/main/resources/mnist`
- `NumKo.kt`: Matrix-/Vektoroperationen für Hot Paths
- `Sigmoid.kt`: Aktivierungsfunktion und Ableitung
- `Main.kt`: Trainingskonfiguration und Programmlauf

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer
- MNIST-Dateien unter `src/main/resources/mnist`

Build ab Projektwurzel:

```bash
mvn -pl chapter-1 test
```

Anwendung starten:

```bash
mvn -pl chapter-1 exec:java
```

Aktuelle Trainingsparameter in `Main.kt`:

| Parameter | Wert |
|-----------|------|
| Layer | `784, 200, 100, 30, 10` |
| Epochen | `30` |
| Mini-Batch-Größe | `64` |
| Worker | `8` |
| Learning Rate | `6.0` |

## Architektur & Abhängigkeiten

Maven-Koordinaten:

```xml
<dependency>
    <groupId>ch.zuegi.machinelearning</groupId>
    <artifactId>learn-neural-networks-chapter-1</artifactId>
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
| Kotlinx Coroutines Core | Parallele Mini-Batch-Verarbeitung |
| Kotlin Deep Learning ONNX | Deep-Learning-Bibliothek im Klassenpfad |
| Kotlin Deep Learning TensorFlow | Deep-Learning-Bibliothek im Klassenpfad |
| Kotlin Deep Learning Dataset | Dataset-Hilfen im Klassenpfad |
| Kotlin Test JUnit 5 | Kotlin-Testintegration |
| JUnit Jupiter | Test-Runner |

Paketstruktur:

```text
src/main/kotlin
├── Main.kt
├── MnistLoader.kt
├── Network.kt
├── NumKo.kt
└── Sigmoid.kt
```

## Konfiguration

Keine `application.yml` vorhanden. Laufzeitparameter stehen aktuell direkt in `Main.kt`.
