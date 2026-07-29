# learn-neural-networks-llm

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin-Modul für LLM-Grundlagen. Aktuell enthält es einen einfachen Regex-Tokenizer mit Vokabularaufbau, Encoding, Decoding und Tests.

## Beschreibung

Das Modul liest den Text `the-verdict.txt` aus den Ressourcen, baut daraus ein Vokabular und wandelt bekannte Eingabetexte in Token-IDs um. Beim Decoding rekonstruiert es Text mit Satzzeichen- und Anführungszeichen-Spacing.

- `MainKt.kt`: Demo-Einstiegspunkt, lädt Textressource und zeigt Encode/Decode
- `SimpleTokenizerV1.kt`: Tokenisierung, Vokabular, Encoding, Decoding
- `SimpleTokenizerV1Test.kt`: Tests für bekannte Tokens, unbekannte Tokens und Roundtrip
- `src/main/resources/text/the-verdict.txt`: Trainings-/Vokabulartext

## Getting Started

Voraussetzungen:

- JDK 8 oder neuer
- Maven 3.8 oder neuer

Tests ausführen:

```bash
mvn -pl llm test
```

Demo ausführen. Die Demo endet aktuell beim eingebauten Unknown-Token-Beispiel mit `Token nicht im Vokabular: 'Hello'`.

```bash
mvn -pl llm exec:java -Dexec.mainClass=ch.zuegi.ml.llm.MainKtKt
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
│   ├── MainKt.kt
│   └── SimpleTokenizerV1.kt
├── main/resources/text
│   └── the-verdict.txt
└── test/kotlin/ch/zuegi/ml/llm
    └── SimpleTokenizerV1Test.kt
```

## Konfiguration

Keine `application.yml` vorhanden. Der Ressourcenpfad ist aktuell in `MainKt.kt` als `/text/the-verdict.txt` festgelegt.
