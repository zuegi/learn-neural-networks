# Kapitel 07: Training und Textgenerierung

**Status:** Umgesetzt und gegen Code/Tests geprüft
**Ziel-Repo:** `zuegi/learn-neural-networks`
**Buchdatei:** `docs/book/kapitel/07-training.adoc` (Anker `[[Training]]`)
**Geltungsbereich:** Bestehender Next-Token-Trainingspfad in Modul `kapitel5`
vom Trainingsfenster bis zum gespeicherten besten Modell und zur
Textgenerierung. Modellarchitektur siehe Kapitel 06.

## Ziel und Zielgruppe

Kapitel 07 erklärt, wie das GPT-Modell aus Kapitel 06 lernt: Daten →
Cross-Entropy-Loss → Backpropagation → Gradient Clipping → Adam →
Validierung → Early Stopping/Checkpoint → Generierung. Jeder Schritt wird mit
Formel und tatsächlichem Codeverhalten verbunden.

## Voraussetzungen

- Kapitel 02 (`TextDataLoader`, `TrainingSample`), Kapitel 04 (Autograd), Kapitel 06 (Forward-Pfad, Logits `[T, V]`).

## Lernziele

1. Next-Token-Ziele als um eins verschobene Sequenz erklären.
2. Loss `L = (1/T) Σ_t −log softmax(z_t)[y_t]` und Gradient `(p_t − onehot(y_t))/T` herleiten; Startwert `≈ ln V` begründen.
3. Batching in `GPTTrainer` inklusive Gradientenakkumulation (Summe über Samples) korrekt beschreiben.
4. `zeroGrad → backward → clip → step` als Reihenfolge begründen.
5. globales L2-Gradient-Clipping formulieren.
6. Adam mit Bias-Korrektur formulieren und den ersten Schritt `≈ lr · sign(g)` erklären.
7. Validierung ohne Dropout und ohne Update, Early Stopping mit `patience`/`minDelta`, Wiederherstellen des besten Checkpoints erklären.
8. Perplexity `exp(mean CE)` interpretieren.
9. Greedy-, Temperatur- und Top-k-Generierung mit Sliding Window erklären.

## Kanonischer Codepfad

| Baustein | Datei |
|---|---|
| Einstieg | `kapitel5/src/main/kotlin/GPTModelTraining.kt` |
| Loss/Generierung | `kapitel5/.../model/GPTModelMultikTensor.kt` (`loss`, `generate`) |
| Trainer | `kapitel5/.../training/GPTTrainer.kt` |
| Optimierer | `kapitel5/.../library/autograd/AdamOptimizer.kt` |
| Early Stopping | `kapitel5/.../training/EarlyStoppingTrainer.kt` |
| Checkpoint | `kapitel5/.../training/ModelCheckpoint.kt` |
| Generierungs-Config | `kapitel5/.../model/GenerationConfig.kt` |
| Daten | `shared/.../TextDataLoader.kt`, `shared/.../TrainingSample.kt` |

`SGDTensorMultik` (Modul `kapitel4`) wird nicht im Trainingspfad verwendet und
nur als Randnotiz erwähnt.

## Verbindliches Codeverhalten (dokumentiert, nicht geändert)

- `GPTTrainer` bildet Batches als zusammenhängende Teilstücke der Sample-Liste, ohne Shuffling; der letzte Batch darf kleiner sein.
- Pro Batch: `zeroGrad()`, dann je Sample `loss(training = true)` und `backward()`. Gradienten addieren sich über den Batch (Summe, keine Division durch die Batchgröße).
- Danach globales Clipping mit `maxGradNorm = 1.0`: `g ← g · c/‖g‖₂`, falls `‖g‖₂ > c`; dann `optimizer.step()`.
- `trainEpoch` liefert den mittleren Sample-Loss, gemessen vor dem jeweiligen Update und mit Dropout.
- `validate` nutzt `training = false`, ruft weder `backward` noch `step` auf.
- Adam: `m ← β1 m + (1−β1) g`, `v ← β2 v + (1−β2) g²`, `m̂ = m/(1−β1^t)`, `v̂ = v/(1−β2^t)`, `θ ← θ − lr · m̂/(sqrt(v̂)+ε)`; Defaults `lr = 1e-3`, `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`; keine Weight Decay.
- Early Stopping: Verbesserung, wenn `val < best − minDelta`; dann Checkpoint schreiben. Abbruch nach `patience` Epochen ohne Verbesserung. Am Ende wird der beste Checkpoint geladen.
- Checkpoint: `Int version (1)`, `Int Anzahl Tensoren`, je Tensor `Int size` + `Double`-Werte in `parameters()`-Reihenfolge. Keine Konfiguration, kein Optimiererzustand.
- Perplexity: `exp(trainLoss)` und `exp(validationLoss)`.
- Generierung: Fenster `takeLast(T)`, letzte Logit-Zeile, `greedy` → argmax; sonst Top-k (`topK in 1 until V`, Gleichstände am Schwellwert bleiben erhalten), Softmax mit Temperatur, inverse CDF-Stichprobe. `startIds.size >= contextLength` ist Pflicht.

## Kapitelstruktur

1. Ziel des Trainings; Pipeline-Diagramm (Mermaid).
2. Trainingsdaten und Train/Validation-Split im Einstiegspunkt.
3. Cross-Entropy-Loss, Gradient, Startwert `ln V`.
4. Batching und Gradientenakkumulation.
5. Backpropagation durch das Modell (Verweis Kapitel 04/06).
6. Gradient Clipping.
7. Adam.
8. Validierung, Early Stopping, Checkpoints.
9. Metriken: Loss und Perplexity.
10. Textgenerierung.
11. Ausführen, Tests, Design-Entscheidungen, Stolpersteine, Zusammenfassung.

## Abgrenzung

- Keine neuen Trainingsfeatures (Shuffling, LR-Schedule, Weight Decay, Mixed Precision, Batch-Tensoren, Optimiererzustand im Checkpoint).
- Keine Änderung der Trainingsalgorithmen; nur Include-Tags und Tests.
- Keine erneute Herleitung von Autograd oder Attention.

## Tests (neu, Modul `kapitel5`)

| Test | Belegt |
|---|---|
| `GPTModelMultikTensorTest` | Logit-Shape, Parameterzahl, Startloss `≈ ln V`, Gradient für jeden Parameter, Kausalität, Generierungslänge/Seed |
| `AdamOptimizerTest` | erster Schritt `≈ lr · sign(g)`, `zeroGrad` |
| `GPTTrainerTest` | Loss sinkt über Epochen, `validate` ändert keine Gewichte |
| `EarlyStoppingTrainerTest` | beste Epoche, Perplexity, Wiederherstellung der besten Gewichte |
| `ModelCheckpointTest` | Save/Load-Roundtrip, Versionsprüfung |

Gradient Clipping ist `private` und wird nicht isoliert getestet; die Formel
wird aus dem Code dokumentiert.

## Messbare Akzeptanzkriterien

- [x] `07-training.adoc` existiert, Anker `[[Training]]`, in `book.adoc` nach Kapitel 06 eingebunden.
- [x] Alle Punkte unter „Verbindliches Codeverhalten“ sind im Kapitel korrekt wiedergegeben.
- [x] Formeln für CE, Gradient, Clipping, Adam und Perplexity sind enthalten.
- [x] Kapitel endet mit Textgenerierung.
- [x] Tests oben existieren und laufen in `mvn verify`.
- [x] Kapitel 05 verweist für Perplexity auf <<Training>>.

## Befunde

- LayerNorm-Gradientenfluss war unterbrochen (Fix siehe Spec Kapitel 06); ohne Fix lernte nur `W_out`.
- Behoben: `kapitel5/pom.xml` konfigurierte `exec-maven-plugin` mit der nicht existierenden `mainClass = MainKt` (`ClassNotFoundException: MainKt`). Jetzt `GPTModelTrainingKt`; `mvn -pl kapitel5 exec:java` startet das Training.
- Train/Validation-Split nimmt die letzten 20 % überlappender Fenster (`stride = 1`); die Grenzfenster teilen Token mit dem Training.

## Quellen und Copyright

Primärquellen: Repository-Code und Tests. Adam (Kingma & Ba, 2015) und
Gradient Clipping werden als Standardverfahren eigenständig formuliert. Keine
fremden Texte oder Codeblöcke.
