# Resultat
Anzahl Zeichen von rawText: 20479
07:51:11.801045 - Start calculation 10 epochs
07:53:26.053633 - epoch 1/10 train=10.5520 val=10.3322 train_ppl=38254.27 val_ppl=30706.65 epoch_time=134s
07:55:41.107804 - epoch 2/10 train=9.4954 val=9.7766 train_ppl=13298.21 val_ppl=17616.87 epoch_time=135s
07:57:58.710930 - epoch 3/10 train=8.4631 val=9.3098 train_ppl=4736.61 val_ppl=11046.24 epoch_time=137s
08:00:16.253824 - epoch 4/10 train=7.5848 val=8.9824 train_ppl=1968.13 val_ppl=7962.02 epoch_time=137s
08:02:40.596643 - epoch 5/10 train=6.8663 val=8.7679 train_ppl=959.42 val_ppl=6424.68 epoch_time=144s
08:04:57.327031 - epoch 6/10 train=6.2911 val=8.6275 train_ppl=539.74 val_ppl=5582.96 epoch_time=136s
08:07:16.411453 - epoch 7/10 train=5.8089 val=8.5409 train_ppl=333.26 val_ppl=5119.92 epoch_time=139s
08:09:33.213053 - epoch 8/10 train=5.4003 val=8.4915 train_ppl=221.48 val_ppl=4873.22 epoch_time=136s
08:11:50.934699 - epoch 9/10 train=5.0604 val=8.4711 train_ppl=157.65 val_ppl=4774.79 epoch_time=137s
08:14:06.823359 - epoch 10/10 train=4.7726 val=8.4696 train_ppl=118.22 val_ppl=4767.50 epoch_time=135s
08:14:07.149712 - Zeit des Trainings: 1375 Sekunden
08:14:07.149884 - best_epoch=10 best_val=8.4696 stopped_early=false
08:14:07.150839 - start:     I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that,
08:14:07.152471 - generated: I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that,ia dazzling.ication

# Analyse
* train fällt stark, val fällt durchgehend bis Epoch 10 (8.4711 -> 8.4696 noch Verbesserung).
* stopped_early=false korrekt: keine val-Stagnation nach deiner minDelta-Regel.
* best_epoch=10 korrekt: bestes val zuletzt.
* generated praktisch gleich wie vorher, erwartbar: gleiche Daten, gleiche Seeds, gleiche Lernkurve.
## Was Ergebnis bedeutet:
* EarlyStopping/Checkpoint funktionieren technisch.
* Aber Setup läuft weiter, weil val noch minimal besser wird.
* Modell weiter unter Datenlimit; Textqualität verbessert langsam.

## Nächste sinnvolle Maßnahmen:
1. Wenn früher stoppen gewünscht: minDelta hochsetzen (0.01 testen) + patience=2.
2. Wenn Qualität wichtig: epochs auf 15-20 erhöhen, aktuelles minDelta lassen.
3. Größter Hebel bleibt Daten: trainingSampleSize erhöhen (oder ganzes Korpus nutzen).
4. Optional Log ergänzen: isBestEpoch pro Zeile ausgeben, damit Checkpoint-Events sichtbar.

## Kurzfazit:
* Kein Bug in EarlyStopping.
* Kein früher Stop, weil val real noch sinkt.
* Nächster Hebel nicht Trainer-Logik, sondern Datenmenge + längeres Training.