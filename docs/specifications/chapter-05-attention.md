# Kapitel 05: Attention

**Status:** Draft  
**Ziel-Repo:** `zuegi/learn-neural-networks`  
**Geltungsbereich:** Spezifikation für Kapitel 05; `docs/book/kapitel/05-attention.adoc` wird in diesem Arbeitsschritt nicht geändert.

## Ziel und Zielgruppe

Kapitel 05 erklärt Scaled Dot-Product Attention und Multi-Head Self-Attention als
trainierbaren Transformer-Baustein. Es richtet sich an Leser, die die
Einbettung einer Token-Sequenz kennen und verstehen, dass Training über
Forward-Pass, Loss und Autograd/Backward läuft.

Das Kapitel soll Konzept, Tensorformen und den tatsächlichen Trainingscode
verbinden. Es darf keine veraltete oder erfundene API als kanonische
Implementierung darstellen.

## Voraussetzungen

- Grundverständnis von Token-, Positions- und Input-Embeddings aus Kapitel 03.
- Grundverständnis von Tensoren, Matrixmultiplikation und Computational Graph/
  Backpropagation aus Kapitel 04.
- Kotlin-Grundkenntnisse.
- Vertrautheit mit Sequenzlänge `ctx`, Embedding-Dimension und einfachen
  linearen Projektionen.
- Keine vorausgesetzte Kenntnis von Multi-Head Attention; die Herleitung erfolgt
  im Kapitel.

## Lernziele

Nach dem Kapitel kann der Leser:

1. die Rollen von Query, Key und Value erklären;
2. Scaled Dot-Product Attention mathematisch herleiten;
3. erklären, warum Scores durch `sqrt(dK)` skaliert werden;
4. Multi-Head Attention anhand von `embeddingDim`, `numHeads`, `dK` und
   `headDim` dimensionieren;
5. die kausale Maske vor dem Softmax einordnen;
6. den Unterschied zwischen Training (`training = true`) und Evaluation
   (`training = false`) beim Attention-Dropout erklären;
7. die Tensorformen entlang des kanonischen Forward-Pfads verfolgen;
8. den didaktischen Scratch-Pfad vom kanonischen, trainierbaren
   `TensorMultik`-Pfad unterscheiden;
9. anhand der Tests Output-Form, Reproduzierbarkeit, Dropout und
   Gradientfluss nachvollziehen.

## Kapitelstruktur

### 1. Motivation und Kontext

Kurz erklären, wie Attention jedem Token kontextabhängige Information aus
anderen Positionen zugänglich macht. Bezugspunkt ist der Input aus Kapitel 03,
nicht eine neue Embedding-Implementierung.

### 2. Query, Key, Value

Für jede Eingabezeile `x` werden gemeinsame lineare Projektionen gebildet:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Die Begriffe Query, Key und Value werden intuitiv und anschließend formal
erklärt. Bias-Optionen des kanonischen Codes dürfen erwähnt werden, bleiben aber
gegenüber dem Attention-Prinzip nachgeordnet.

### 3. Scaled Dot-Product Attention

Für eine Head-Dimension `dK`:

```text
scores = Q K^T / sqrt(dK)
weights = softmax(scores)
headOutput = weights V
```

Softmax wird zeilenweise über die Key-Positionen erklärt. Bei kausaler
Attention werden Einträge für `j > i` vor dem Softmax auf `-∞` gesetzt. Damit
kann Position `i` keine zukünftige Position sehen.

### 4. Multi-Head-Aufbau

Die projizierten Matrizen werden entlang der Spalten in `numHeads` Köpfe
aufgeteilt. Jeder Kopf berechnet eigene Scores und gewichtete Values. Die
Head-Ausgaben werden wieder konkateniert und durch `W_output` auf
`embeddingDim` projiziert.

### 5. Kanonische Kotlin-Implementierung

Hauptreferenz ist:

`kapitel4/src/main/kotlin/ch/zuegi/ml/llm/kapitel4/library/autograd/MultiHeadAttentionMultikTensor.kt`

Die dokumentierte API lautet:

```kotlin
forward(input: TensorMultik, ctx: Int, training: Boolean = false): TensorMultik
```

Der Text muss den tatsächlichen Ablauf der Klasse abbilden:

1. Größenprüfung von `input`;
2. Q/K/V-Projektion inklusive optionaler Bias-Tensoren;
3. Aufteilung in `numHeads` Köpfe;
4. Score-Berechnung, Skalierung und optionale kausale Maske;
5. zeilenweiser Softmax;
6. Attention-Dropout nur bei `training == true`;
7. gewichtete Value-Summe je Position und Head;
8. Konkatenation der Köpfe;
9. Output-Projektion inklusive optionalem Output-Bias.

`parameters()` und der Rückfluss über `out.backward()` werden als Teil des
trainierbaren Pfads erklärt, ohne eine zweite Attention-API einzuführen.

### 6. Tests und typische Fehler

Tests dienen als ausführbare Spezifikation für Output-Größe,
Seed-Reproduzierbarkeit, Trainings-Dropout, Parameter-/Bias-Registrierung und
Gradienten. Typische Fehler: falsche Matrixdimension, falsche Softmax-Achse,
Maskierung nach statt vor Softmax, versehentlich aktiver Dropout in Evaluation
und Verwechslung von `dK` mit `headDim`.

## Mathematische Tiefe

Die Tiefe ist anwendungsorientiert mit vollständiger Shape- und
Operationsherleitung:

- Skalarprodukt `q_i · k_j`;
- Skalierung `1 / sqrt(dK)`;
- zeilenweiser Softmax;
- gewichtete Summe über Values;
- Konkatenation und Output-Projektion;
- qualitative Erklärung des Gradientenflusses durch Matrixmultiplikation und
  Softmax.

Eine vollständige Herleitung der Softmax-Jacobi-Matrix, Optimierer oder
alternativer Attention-Varianten gehört nicht in dieses Kapitel. Sie kann als
Verweis auf Kapitel 04 bzw. Exkurs dienen.

## Verbindliche Tensor-Shape-Regeln

Alle Formen beziehen sich auf row-major `TensorMultik`-Daten:

| Bezeichnung | Form | Bedeutung |
|---|---:|---|
| `input` | `[ctx, embeddingDim]` | Eingabezeilen pro Token |
| `W_Q`, `W_K`, `W_V` | `[embeddingDim, headDim]` | gemeinsame Projektionen |
| `Q`, `K`, `V` | `[ctx, headDim]` | projizierte Sequenz |
| `headDim` | `numHeads * dK` | Gesamtbreite aller Köpfe |
| `Q_h`, `K_h`, `V_h` | `[ctx, dK]` | ein Kopf |
| `scores_h` | `[ctx, ctx]` | Query-Position gegen Key-Position |
| `weights_h` | `[ctx, ctx]` | zeilenweise normalisierte Scores |
| `headOutput_h` | `[ctx, dK]` | gewichtete Value-Summen |
| konkatenierte Köpfe | `[ctx, headDim]` | Rückführung aller Köpfe |
| `W_output` | `[headDim, embeddingDim]` | Output-Projektion |
| Rückgabe | `[ctx, embeddingDim]` | Attention-Ausgabe |

Für jeden Forward-Aufruf muss `input.size == ctx * embeddingDim` gelten.
`ctx` ist Sequenzlänge, nicht `embeddingDim`, `dK` oder Anzahl Köpfe.

## Codepfade: klare Trennung

### Kanonischer Trainingspfad

Der einzige Hauptpfad des Kapitels ist
`library/autograd/MultiHeadAttentionMultikTensor.kt` mit
`TensorMultik`. Er ist trainierbar, registriert Parameter und wird durch Tests
unter
`kapitel4/src/test/kotlin/ch/zuegi/ml/llm/kapitel4/library/autograd/MultiHeadAttentionMultikTensorTest.kt`
abgesichert. Kapitel 05 muss diesen Pfad für Integrations- und
Trainingsbeispiele verwenden.

### Didaktischer Scratch-Pfad

`kapitel4/src/main/kotlin/ch/zuegi/ml/llm/kapitel4/scratch/autograd/Tensor.kt`
und
`kapitel4/src/main/kotlin/ch/zuegi/ml/llm/kapitel4/scratch/autograd/MultiHeadAttention.kt`
zeigen dieselbe Idee mit einer kleineren, didaktischen Tensorabstraktion.
Sie dürfen für Intuition, einzelne Rechenschritte und Scratch-Tests erwähnt
werden. Sie sind kein gleichwertiger Trainingspfad, keine normative API und
keine Grundlage für neue Kapitel-05-Integrationen.

## Abgrenzung

Nicht Bestandteil:

- Änderung oder Umbenennung von `05-attention.adoc`;
- Implementierung neuer Attention-Klassen;
- vollständiger Transformer-Block inklusive Feed-Forward, LayerNorm oder
  Residual-Verbindungen;
- Tokenisierung und Embedding-Implementierung;
- GQA, MQA, Flash Attention oder Encoder-Decoder-Cross-Attention;
- Optimierer, Batching über mehrere Sequenzen und Performance-Benchmarking;
- parallele Vollbeschreibung von Scratch- und Library-Code.

Die vorhandene AsciiDoc-Datei enthält eine veraltete Pseudocode-Signatur und
verweist mit `:sourceDirKapitel2` fälschlich auf `kapitel3`. Diese Spec
dokumentiert den Korrekturbedarf; die AsciiDoc-Änderung ist ein nachgelagerter
Umsetzungsschritt.

## Kleine Umsetzungstasks

| ID | Aufgabe | Abhängigkeiten | Datei-/Testbezug |
|---|---|---|---|
| ATT-01 | AsciiDoc-Header und Include-Variablen auf Kapitel-04-/kanonische Pfade korrigieren | keine | `docs/book/kapitel/05-attention.adoc` |
| ATT-02 | Veraltete Pseudocode-API durch tatsächliche `MultiHeadAttentionMultikTensor`-Signatur und Ablauf ersetzen | ATT-01 | kanonische Kotlin-Datei |
| ATT-03 | Shape-Tabelle und Q/K/V-/Head-Herleitung einbauen | ATT-02 | `MultiHeadAttentionMultikTensor.kt` |
| ATT-04 | Kausale Maske vor Softmax und Dropout nur im Training erklären | ATT-02 | `forwardHead`, Scratch-Causal-Tests |
| ATT-05 | Scratch-Pfad als optionalen Lernpfad kennzeichnen, ohne parallele Hauptdoku | ATT-02 | `scratch/autograd/Tensor.kt`, `MultiHeadAttention.kt` |
| ATT-06 | Testbeispiele bzw. Verweise für Output, Seed, Dropout, Parameter und Gradienten ergänzen | ATT-02, ATT-03 | `MultiHeadAttentionMultikTensorTest.kt` |
| ATT-07 | Kapitelreferenzen und angrenzende Kapitel auf konsistente Begriffe prüfen | ATT-01–ATT-06 | Kapitel 03, Kapitel 04, `docs/HowToOrganiseBook.md` |
| ATT-08 | Zeilenweise Softmax-Summen, maskierte Zukunft (`0`) sowie `ctx = 1` und `ctx = 2` als Edge Cases testen | ATT-04, ATT-06 | Library-/Scratch-Causal-Tests |
| ATT-09 | Shape-Invarianten pro Head und Verhältnis `dK`/`headDim` explizit dokumentieren; Verhalten bei `embeddingDim != numHeads * dK` festlegen | ATT-03 | Shape-Regeln, kanonische Kotlin-Datei |
| ATT-10 | Zentrale Finite-Difference-Prüfung für Autograd als Validierungs-Backlog ergänzen | ATT-06 | `MultiHeadAttentionMultikTensorTest.kt` |
| ATT-11 | Numerische Softmax-Stabilität mit Wert `1000` und maskiertem `-Inf` spezifizieren und testen | ATT-04, ATT-08 | Softmax-/Causal-Test |
| ATT-12 | Perplexity `= exp(mean CE)` nur bei ausreichendem Kapitelumfang als optionalen Anschluss aufnehmen | ATT-07 | Kapitelumfang/Trainingsabschnitt |

Reihenfolge: `ATT-01 → ATT-02 → ATT-03/ATT-04/ATT-05 → ATT-06 → ATT-07 → ATT-08/ATT-09/ATT-10/ATT-11/ATT-12`.

## Messbare Akzeptanzkriterien

- [ ] `docs/book/kapitel/05-attention.adoc` bleibt in diesem Spec-Change
      unverändert; diese Datei ist die einzige neue Datei.
- [ ] Kapiteltext verwendet als kanonische API exakt
      `forward(input: TensorMultik, ctx: Int, training: Boolean = false)`.
- [ ] `:sourceDirKapitel2` zeigt nach Umsetzung nicht mehr auf `kapitel3`;
      Ziel und relative Pfade sind auf den tatsächlichen Referenzpfad geprüft.
- [ ] Alle neun Shape-Regeln der Tabelle sind im Kapitel oder gleichwertig
      enthalten; insbesondere `scores/weights = [ctx, ctx]`.
- [ ] Causal Mask wird explizit vor Softmax beschrieben.
- [ ] Zeilenweise Softmax-Summen ergeben `1`; maskierte Zukunftspositionen ergeben
      nach Softmax `0`.
- [ ] Tests decken `ctx = 1` und `ctx = 2` ohne Off-by-one-Fehler ab.
- [ ] Dropout wird explizit auf `training == true` begrenzt.
- [ ] Shape-Invarianten gelten pro Head; `dK` und `headDim = numHeads * dK`
      werden nicht synonym verwendet.
- [ ] Verhalten bei `embeddingDim != numHeads * dK` ist als erlaubt und durch
      Output-Projektion abgedeckt dokumentiert.
- [ ] Eine zentrale Finite-Difference-Prüfung validiert mindestens ausgewählte
      Autograd-Gradienten.
- [ ] Softmax-Verhalten bleibt bei Scores um `1000` und maskierten `-Inf`
      numerisch stabil.
- [ ] Kanonischer Library-Pfad und didaktischer Scratch-Pfad sind in getrennten
      Abschnitten mit unterschiedlichen Rollen beschrieben.
- [ ] Es gibt genau einen hervorgehobenen Hauptpfad für Training.
- [ ] Verweise auf mindestens einen Test für Outputform, Dropout und
      Gradienten sind vorhanden.
- [ ] Kapitel 03, Kapitel 04 und `docs/HowToOrganiseBook.md` widersprechen der
      Pfad- und Rollenbeschreibung nicht.

## Quellen und Copyright

Primärquellen sind die im Repository vorhandenen Kotlin-Dateien,
zugehörigen Tests sowie die bestehenden Kapitel- und Strukturhinweise.
Mathematische Standardnotation für Attention wird eigenständig formuliert.

Keine fremden Codeblöcke oder längeren wörtlichen Übernahmen einfügen.
Fremde Quellen, Abbildungen oder Buchbeispiele nur mit nachvollziehbarer
Quellenangabe und kompatibler Lizenz verwenden. Repository-interne
Codeausschnitte bleiben auf notwendige, kurze Lehrbeispiele begrenzt und
verlinken auf die Originaldatei.

## Risiken und offene Punkte

- Die endgültige relative AsciiDoc-Pfadstruktur muss beim ATT-01-Umsetzungsschritt
  gegen die tatsächliche Buchstruktur geprüft werden.
- Die aktuelle Implementierung nutzt einen expliziten `ctx`-Parameter und
  flache row-major Tensoren; eine spätere Shape-API darf nicht stillschweigend
  in dieses Kapitel zurückprojiziert werden.
- Verhalten und didaktische Erwartung der Scratch-Causal-Tests können sich
  unterscheiden; Scratch-Tests dürfen daher keine Library-Verträge ersetzen.
- Bias-Flags, Seeds und Dropout sind Implementierungsdetails mit Testabdeckung,
  aber keine Voraussetzung für die mathematische Einführung.
- Ob Kapitel 05 zusätzlich einen vollständigen Trainingsloop zeigt, bleibt
  offen und muss mit der Kapitel-05-Gesamtstruktur entschieden werden; der
  Attention-Forward-/Backward-Vertrag ist unabhängig davon verbindlich.
- Perplexity (`exp(mean CE)`) bleibt optional und wird nur aufgenommen, wenn
  Trainingsmetriken im finalen Kapitelumfang enthalten sind.
