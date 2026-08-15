# Wie schreibe ich die Doku

## Werkzeug für das Buch/Doku
Ich werde für die Doku AsciiDoc verwenden

## Repo-Struktur
Ziel ist es die Doku vom Code zu trennen, aber doch in demselben Repo zu verwalten.
```text
learn-neural-networks/
├── src/
├── docs/
│   └── book/
│       ├── book.adoc              # Hauptdatei mit includes
│       ├── kapitel/
│       │   ├── 00-einleitung.adoc
│       │   ├── 01-tokenisierung.adoc
│       │   ├── 02-embeddings.adoc
│       │   ├── 03-attention.adoc
│       │   └── ...
│       └── images/
└── build.gradle.kts               # asciidoctor-Plugin hier registrieren
```

## Kapitelstruktur
Jedes Kapitel verbindet das Konzept mit dem Code anstatt nur die reine Theorie zu wiederholen.
1. Konzept kurz erklären (Was macht dieser Baustein einer LLM und warum)
2. Kotlin implementieren zeigen über den direkten Link auf den entsprechenden Code im Repo ([siehe Attention.kt](../../src/.../Attention.kt))
3. Design Entscheidungen dokumentieren. Wo bin ich bewusst von der gängigen Lehrpraxis/Lehrbuch abgewichen, was habe ich vereinfacht, welcher Kotlin Idiome genutzt?
4. Setze Diagrame (Mermaid) für die Veranschaulichung von Modellen ein.
5. Stolpersteine / Learnings am Ende eines Kapitels einfügen. Das ist der wortvollste Teil für mich selbst und andere

## Praktisches Vorgehen
1. Ordnerstruktur anlegen und SUMMARY (= Index) erstellen
2. Ein Kapitel pro Baustein der LLM Pipeline schreiben, in der Reihenfolge, in der die Daten durchlaufen (Tokenizer -> Embedding -> Attention....-> Output)
3. Code Referenzen als relative Links einbauen, damit sie auch nach Code und Buch Refactorings über git grep leicht auffindbar sind
4. 