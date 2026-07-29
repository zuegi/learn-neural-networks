# learn-neural-networks-llm

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin-Modul für spätere LLM-Experimente innerhalb des `learn-neural-networks` Multi-Module-Projekts.

## Beschreibung

Das Modul ist aktuell bewusst leer angelegt. Es enthält Maven- und Kotlin-Build-Konfiguration, aber noch keine fachliche Implementierung.

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer

Build ab Projektwurzel:

```bash
mvn -pl llm test
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

Paketstruktur:

```text
src/main/kotlin/ch/zuegi/ml/llm
└── LlmModule.kt
```

## Konfiguration

Keine `application.yml` vorhanden.
