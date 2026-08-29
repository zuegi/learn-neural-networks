# Fahrplan Kapitel 3-5

## Zielbild
- Didaktischer Pfad bleibt in `kapitel3` und `kapitel4/scratch`.
- Kanonischer Ausfuehrungspfad liegt in `shared` + `kapitel4/library/autograd`.
- `kapitel5` enthaelt nur Training/Anwendung und konsumiert kanonische Klassen.

## Ist-Stand (2026-08-28)

### Kapitel 3
- Inhalt und Doku vorhanden (`docs/book/kapitel/03-embeddings.adoc`).
- Tests vorhanden.
- Rolle im Fahrplan: didaktische Herleitung + Vergleich zu kanonischem Pfad.

### Kapitel 4
- Kanonische Autograd-Bausteine liegen in `kapitel4/library/autograd`.
- Scratch-Pfad vorhanden (`kapitel4/scratch/autograd`).
- Doku in `docs/book/kapitel/04-attention.adoc` noch duenn und ausbaufaehig.

### Kapitel 5
- Maven-Dependency auf `kapitel4` ist gesetzt.
- Training laeuft (siehe `kapitel5/ongoing.md`).
- `Trainable*Embedding` wurde in `shared` aufgenommen (kanonischer Ablageort).
- Testabdeckung in `kapitel5` aktuell schwach bzw. fehlt.

## Leitplanken
- Keine lokalen Duplikate fuer Embedding/Tokenizer in `kapitel5`.
- Imports in `kapitel5` zeigen auf `shared` und `kapitel4/library/autograd`.
- Architekturregel: didaktischer Pfad != kanonischer Pfad.
- Jede Migration endet mit gruenem Modul-Build.

## Arbeitsplan

### Phase A - Konsolidierung Imports und Duplikate (kurzfristig)
- [ ] Alle verbleibenden Imports in `kapitel5` auf kanonische Pfade umstellen.
- [ ] Lokale Duplikate unter `kapitel5/library/autograd` und `kapitel5/library/tokenize` entfernen, wenn ersetzt.
- [ ] Pruefen, ob in `kapitel5/library/embedding` noch redundante Klassen liegen; falls ja entfernen.
- [ ] Build-Check: `kapitel5` kompiliert ohne lokale Duplikat-Abhaengigkeit.

### Phase B - Tests fuer Kapitel 5 (kurzfristig)
- [ ] Unit-Tests fuer `EarlyStoppingTrainer` (minDelta, patience, bestEpoch).
- [ ] Unit-Tests fuer `ModelCheckpoint` (save/load, best-only Verhalten).
- [ ] Unit-Tests fuer `GPTTrainer` (Loss-/PPL-Tracking, early-stop Trigger).
- [ ] Kleiner Integrationstest: 1-2 Epochen Smoke-Training mit kleinem Sample.

### Phase C - Doku Kapitel 4/5 (mittelfristig)
- [x] Neues Kapitel-4-Dokument anlegen: `04-autograd.adoc` (Tensor-Klasse, Forward/Backward, Konzept).
- [x] `docs/book/kapitel/05-attention.adoc` erstellen (Q/K/V, Masking, Multi-Head).
- [x] `docs/book/book.adoc` mit neuen Includes aktualisieren.
- [ ] Kapitel-6-Dokument für Training Loop/Early Stopping/Checkpoints (fortgesetzt in Phase D).
- [ ] Cross-Links 3 -> 4 -> 5 -> 6 in den Texten setzen.

### Phase D - Abschluss und Qualitaetstor (mittelfristig)
- [ ] Reactor-Build fuer `shared`, `kapitel3`, `kapitel4`, `kapitel5` gruen.
- [ ] Keine Referenzen mehr auf geloeschte lokale Duplikate.
- [ ] Smoke-Run fuer Training dokumentieren (Parameter + Ergebnis).
- [ ] Kurzprotokoll der finalen Struktur in `ReStrukturierung.md` nachziehen.

## Definition of Done (Kapitel 3-5)
- [ ] `kapitel5` nutzt Embeddings/Tokenizer nur aus `shared`.
- [ ] `kapitel5` nutzt Autograd-Bausteine nur aus `kapitel4/library/autograd` (oder bewusst begruendete Ausnahmen).
- [ ] Test-Suite in `kapitel5` deckt Kernpfad Training + Checkpoint + EarlyStopping ab.
- [ ] Buch-Doku enthaelt Kapitel 3, 4 und 5 als konsistente Hauptstrecke.
- [ ] Build/Test fuer betroffene Module erfolgreich.

## Offene Entscheidungen
- [ ] Soll `AdamOptimizer` langfristig in kanonischen Pfad (`kapitel4/library/autograd` oder `shared`) wandern?
- [ ] Welche minimale Testlaufzeit ist fuer CI akzeptabel (z. B. < 60s fuer kapitel5-Smoke-Tests)?
- [ ] Welche Trainingsartefakte werden versioniert, welche lokal ignoriert?

## Verifikation (manuell)
```bash
cd "/Users/groot/WS/zuegi/machine-learning/learn-neural-networks"
mvn -pl kapitel5 -am test
mvn -pl shared,kapitel3,kapitel4,kapitel5 -am test
```

