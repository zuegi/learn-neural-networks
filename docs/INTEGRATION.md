# Integration ins bestehende Maven-Projekt

## 1. Dateien übernehmen

Kopiere aus diesem Grundgerüst in dein Repo (auf oberster Ebene, neben `../pom.xml`):

```
docs/book/            -> dein-projekt/docs/book/
```

`pom-plugin-snippet.xml` und `INTEGRATION.md` bleiben nur zur Referenz, müssen
nicht mit ins Repo übernommen werden.

## 2. Plugin in die pom.xml einfügen

Öffne deine `../pom.xml` und füge den Inhalt von `../../../../../Downloads/files/pom-plugin-snippet.xml` in den
`<plugins>`-Block innerhalb von `<build>` ein.

Falls dir die PDF-Generierung egal ist: die zweite `<execution>`
(`generate-book-pdf`) und den `<dependencies>`-Block darunter einfach löschen –
dann brauchst du nur die HTML-Execution.

## 3. Buch bauen

```bash
mvn generate-resources
```

Das erzeugte Buch liegt danach unter:

```
target/generated-docs/book.html
target/generated-docs/book.pdf   (falls PDF-Execution aktiv)
```

Öffne `book.html` einfach im Browser.

## 4. Code-Include-Pfade anpassen

In `book/kapitel/03-attention.adoc` ist als **Beispiel** ein Include auf
`../../../src-example/Attention.kt` eingebaut (liegt in diesem Grundgerüst
unter `src-example/`). Das ist nur eine Demo-Datei.

Sobald dein echtes Projekt vorliegt:

1. `src-example/` kannst du löschen
2. Passe den `include::`-Pfad in `03-attention.adoc` auf deine echte Datei an,
   z. B.:

   ```
   include::../../../src/main/kotlin/llm/attention/Attention.kt[tags=attention-score]
   ```

   Der Pfad ist relativ zur `.adoc`-Datei in `book/kapitel`.

3. Setze in deiner echten Kotlin-Datei die passenden Tag-Marker:

   ```kotlin
   // tag::attention-score[]
   fun computeAttentionScore(...) { ... }
   // end::attention-score[]
   ```

4. Wende das gleiche Muster auf die übrigen Kapitel-Stubs an (01, 02, 04, 05,
   06) – dort stehen `// TODO`-Kommentare als Platzhalter für Konzepttext und
   Code-Includes.

## 5. Optional: GitHub Pages Deployment

Um `target/generated-docs/book.html` automatisch bei jedem Push zu
veröffentlichen, kannst du eine GitHub Actions Workflow-Datei
`.github/workflows/docs.yml` anlegen, die `mvn generate-resources` ausführt
und `target/generated-docs/` per `actions/deploy-pages` published. Sag
Bescheid, falls du dafür ebenfalls ein Grundgerüst möchtest.

## 6. Kapitel-Reihenfolge ändern / neue Kapitel hinzufügen

Alle Kapitel werden zentral in `book/book.adoc` per `include::` in der
gewünschten Reihenfolge eingebunden. Neues Kapitel = neue `.adoc`-Datei in
`book/kapitel` anlegen + eine `include::`-Zeile in `book/book.adoc` ergänzen.
