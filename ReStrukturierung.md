# ReStrukturierung – Stand und Entscheidungen

## 1) Zielarchitektur: Zwei klare Pfade

- **Didaktischer Pfad**: `kapitel3`, `kapitel4/scratch`, `shared/embedding/Trainable*`
  - Fokus: Herleitung, Verständnis, Micrograd/Autograd-Prinzipien, manueller Backward-Pass.
- **Kanonischer Ausführungspfad**: `shared` + `kapitel4/library/autograd`
  - Fokus: trainierbarer, wartbarer Code auf Basis von `TensorMultik`.
- **`kapitel5`** enthält GPT-Modell, Training und Demos und nutzt nur den kanonischen Pfad.

### Warum

- weniger Duplikate (Tokenizer/Embeddings/Modelle)
- weniger Doku-Drift
- Bugfixes nur einmal
- klarer Übergang von Lernen -> Anwenden

## 2) Modulrollen (Ist-Stand)

| Modul | Inhalt | Abhängigkeiten |
|---|---|---|
| `shared` | Tokenizer (`SimpleTokenizerV1`, `GPT2Tokenizer`), `TextDataLoader`, Embeddings ohne Autograd und `Trainable*`-Embeddings | – |
| `kapitel2` | nur Demos zu Buchkapiteln 01–03 | `shared` |
| `kapitel3` | Attention ohne Autograd (scratch + Multik) | `shared` |
| `kapitel4` | `scratch/autograd` (didaktisch), `library/autograd` (kanonisch, inkl. `EmbeddingMultikTensor`) | `shared` |
| `kapitel5` | GPT-Modelle, `GPTConfig`, `GenerationConfig`, Training, Adam, Demos | `shared`, `kapitel4` |

Maven-Modulnummern und Buchkapitel sind nicht identisch (z. B. `kapitel5` → Buchkapitel 06 und 07).

## 3) Entscheidung Embeddings (Variante A)

- `TrainableTokenEmbedding`/`TrainablePositionalEmbedding`/`TrainableInputEmbedding` rechnen den Backward-Pass
  von Hand und aktualisieren per SGD. Sie sind mit dem Autograd-Graphen von `TensorMultik` nicht kompatibel
  und bleiben die didaktische Version aus Buchkapitel 03.
- Das GPT-Modell nutzt `kapitel4/library/autograd/EmbeddingMultikTensor` (eine Klasse für Token- und
  Positions-Embedding). Damit behandeln Adam, Gradient-Clipping und Checkpoints die Embeddings wie alle
  anderen Parameter.

## 4) Migrationsplan – erledigt

1. **Phase 0 – Training nachweisen:** `TrainingComparisonDemo` vergleicht „nur W_out“, „Embeddings statisch“
   und „alle Parameter“; trainierbare Gewichte senken den Validierungs-Loss deutlich. Clipping berücksichtigt
   nur Optimizer-Parameter.
2. **Phase 1 – Duplikate entfernen:** codeidentische Embedding-/Attention-Kopien in `kapitel2`, `kapitel4`,
   `kapitel5` gelöscht; alle GPT-Modellklassen nach `kapitel5` konsolidiert.
3. **Phase 2 – Tokenizer zentralisieren:** dokumentierte `SimpleTokenizerV1` nach `shared`;
   `R50kBpeTokenizer` durch `shared/tokenize/GPT2Tokenizer` ersetzt.
4. **Phase 3 – Autograd-Embedding:** `EmbeddingMultikTensor` aus `GPTModelMultikTensor` extrahiert
   (Loss, Parameter und Gradienten bitidentisch).
5. **Phase 4 – Doku:** Includes/Links geprüft, Specs, `HowToOrganiseBook.md` und dieses Dokument aktualisiert.

## Akzeptanzkriterien

- [x] `kapitel2/3/4/5` bauen ohne lokale Duplikat-Abhängigkeiten (`mvn clean verify`).
- [x] Alle AsciiDoc-Includes/Links zeigen auf existierende, nicht duplizierte Dateien.
- [ ] Doku referenziert pro Thema genau einen Hauptpfad: erfüllt für Kapitel 01–03 und 05–07 (Kapitel 05 markiert
      Scratch als „Optionaler Lernpfad“); Kapitel 04 baut bewusst auf dem didaktischen `scratch`-Pfad auf –
      ob `TensorMultik` dort Hauptpfad werden soll, ist offen.
- [x] GPT-Training läuft vollständig über Autograd-Parameter (`EmbeddingMultikTensor`, `TransformerBlockMultikTensor`,
      `LayerNormMultikTensor`, `W_out`); `Trainable*` bleibt bewusst didaktisch (Variante A).
- [x] Änderungen an Tokenizer-/Embedding-Klassen erfolgen nur noch an einer Stelle.

## Offen

- `kapitel3` (Attention ohne Autograd) wird vom Buch derzeit nicht referenziert.
- Gleichnamige Klassen `MultiHeadAttention` in `kapitel3/scratch` und `kapitel4/scratch/autograd` sind
  bewusst verschieden (ohne bzw. mit Autograd), kein Duplikat.
