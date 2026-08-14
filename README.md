# learn-neural-networks

<!-- TODO: Beschreibung in pom.xml ergänzen -->

Kotlin/Maven Multi-Module-Projekt zum Lernen von neuronalen Netzen und LLM-Grundlagen. 

## Was sind die nächsten Schritte
1. Code technisch weiterführen (jetzt)
* Eval-Loop sauber: val-loss, perplexity, best-checkpoint, early-stop.
* Reproduzierbarkeit: Seed, Config-Snapshot, Run-Logs.
* Save/Load robust testen (Roundtrip-Test, Shape-Checks, Fehlerfälle).

2. Doku aufbauen (parallel ab jetzt, intensiv danach)
* Kapitel 1..5: pro Kapitel Ziel, Kernideen, wichtigste Klassen, Datenfluss.
* Fokus auf “wie Training wirklich läuft” statt nur API-Liste.
* Kleine Runbooks: “Train starten”, “Checkpoint laden”, “Metriken lesen”.

3. Pretrained-Weights später, aber realistisch
* Erst mit offenem Checkpoint testen (z. B. kleines GPT2-kompatibles Modell)
* Danach Import-Adapter bauen (name mapping + tensor reshape + validation script).