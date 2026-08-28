# ReStrukturierung – Zusammenfassung der letzten 2 Empfehlungen

## 1) Zielarchitektur: Zwei klare Pfade

- **Didaktischer Pfad**: `kapitel3` + `kapitel4/scratch`
  - Fokus: Herleitung, Verständnis, Micrograd/Autograd-Prinzipien.
- **Kanonischer Ausführungspfad**: `shared` + `kapitel4/library`
  - Fokus: trainierbarer, wartbarer, produktionsnaher Code.
- **Kapitel 5** soll nur noch Trainings-/Anwendungsmodul sein und den kanonischen Pfad nutzen.

### Warum

- weniger Duplikate (Tokenizer/Embeddings/Modelle)
- weniger Doku-Drift
- Bugfixes nur einmal
- klarer Übergang von Lernen -> Anwenden

---

## 2) Pragmatischer Migrationsplan (in Reihenfolge)

### Phase 1 – Zentralisieren

1. Tokenizer nach `shared`:
   - `SimpleTokenizerV1`
   - `GPT2Tokenizer`
2. Non-Autograd-Embeddings nach `shared`:
   - Token/Positional/Input (Array + Multik, je nach Bedarf)
3. Imports in `kapitel3` und `kapitel4` auf `shared` umstellen.

### Phase 2 – Kapitel 4 säubern

4. `kapitel4` klar trennen:
   - `scratch/autograd` = didaktisch
   - `library/autograd` = kanonisch
5. Doppelte Embedding-/Tokenizer-Kopien aus `kapitel4` entfernen.

### Phase 3 – Kapitel 5 auf kanonischen Pfad binden

6. `kapitel5` auf `kapitel4/library` + `shared` referenzieren (keine lokalen Duplikate).
7. Falls nötig Maven-Dependency `kapitel5 -> kapitel4` setzen.

### Phase 4 – Doku konsolidieren

8. AsciiDoc-Includes auf kanonische Dateien zeigen lassen.
9. Didaktikpfad im Text als optionalen Lernpfad markieren.
10. Für Kapitel 4/5 primär den trainierbaren Hauptpfad dokumentieren.

---

## Akzeptanzkriterien

- `kapitel2/3/4/5` bauen ohne lokale Duplikat-Abhängigkeiten.
- Doku referenziert pro Thema genau einen Hauptpfad.
- Training läuft über `TrainableTokenEmbedding` + `TrainablePositionalEmbedding` (+ `TrainableInputEmbedding`).
- Änderungen an Tokenizer/Embedding-Klassen müssen nur noch an einer Stelle erfolgen.

