# Kapitel 06: GPT-Modell (Decoder-only Transformer)

**Status:** Umgesetzt und gegen Code/Tests geprüft
**Ziel-Repo:** `zuegi/learn-neural-networks`
**Buchdatei:** `docs/book/kapitel/06-gpt.adoc` (Anker `[[GPT]]`)
**Geltungsbereich:** Forward-Pfad des bestehenden GPT-Modells von Token-IDs bis
zu Vokabular-Logits. Training, Loss-Optimierung und Textgenerierung gehören in
Kapitel 07.

## Ziel und Zielgruppe

Kapitel 06 setzt die Bausteine aus Kapitel 03 (Embeddings), 04 (Autograd) und
05 (Multi-Head Attention) zu einem decoder-only GPT-Modell zusammen. Leser
sollen jede Operation mathematisch und mit Tensorform nachvollziehen und im
Kotlin-Code wiederfinden können.

## Voraussetzungen

- Kapitel 03: Token- und Positions-Embeddings.
- Kapitel 04: `TensorMultik`, Computational Graph, `backward()`.
- Kapitel 05: `MultiHeadAttentionMultikTensor` inklusive kausaler Maske.

## Lernziele

Nach dem Kapitel kann der Leser:

0. erklären, wofür "GPT" steht (Generative Pre-trained Transformer) und welches
   konkrete Modell (Architektur-Vorbild GPT-2, tatsächliche Konfiguration in
   `GPTModelTraining.kt` vs. `GPTConfig`-Defaults) dieses Buch implementiert;
1. die Konfiguration `GPTConfig` und die abgeleitete Größe `dK = embeddingDim / numHeads` erklären;
2. die Eingabematrix `X0 = E_tok[ids] + E_pos[0..T-1]` bilden;
3. einen Pre-LN-Transformer-Block als `A = X + MHA(LN1(X))`, `Y = A + FFN(LN2(A))` formulieren;
4. LayerNorm, GELU-FFN und Residualverbindungen mathematisch beschreiben;
5. die finale LayerNorm und die Projektion auf `vocabSize` Logits erklären;
6. alle Tensorformen entlang des Forward-Pfads angeben;
7. die Parameterzahl des Modells aus der Konfiguration berechnen;
8. begründen, warum die kausale Maske (`causal = true`, Default) frühere Logits unabhängig von späteren Token macht.

## Kanonischer Codepfad

| Baustein | Datei |
|---|---|
| Modell | `kapitel5/src/main/kotlin/ch/zuegi/ml/llm/kapitel5/model/GPTModelMultikTensor.kt` |
| Konfiguration | `kapitel5/src/main/kotlin/ch/zuegi/ml/llm/kapitel5/model/GPTConfig.kt` |
| Embeddings | `kapitel4/.../library/autograd/EmbeddingMultikTensor.kt` |
| Transformer-Block | `kapitel4/src/main/kotlin/ch/zuegi/ml/llm/kapitel4/library/autograd/TransformerBlockMultikTensor.kt` |
| Attention (Kapitel 05) | `kapitel4/.../library/autograd/MultiHeadAttentionMultikTensor.kt` |
| Feed-Forward | `kapitel4/.../library/autograd/FeedForwardMultikTensor.kt` |
| LayerNorm | `kapitel4/.../library/autograd/LayerNormMultikTensor.kt` → `TensorMultik.layerNorm` |

Kapitel 06 dokumentiert das Modell in `kapitel5`, das in Kapitel 07
trainiert wird. Die frühere Vorgängerkopie in `kapitel4` wurde entfernt;
alle GPT-Modellklassen liegen in `kapitel5/model`.

## Verbindliche Formeln und Shapes

Notation: `T = contextLength`, `d = embeddingDim`, `V = vocabSize`,
`h = numHeads`, `dK = d / h`, `H = hiddenDim`, `L = numLayers`. Matrizen sind
row-major, eine Zeile pro Position.

| Größe | Form | Definition |
|---|---:|---|
| `tokenIds` | `[T]` | exakt `contextLength` IDs |
| `E_tok` | `[V, d]` | `tokenEmbedding` (`EmbeddingMultikTensor`), Init `N(0, 0.01²)` |
| `E_pos` | `[T, d]` | `positionalEmbedding` (`EmbeddingMultikTensor`), Init `N(0, 0.01²)` |
| `X0` | `[T, d]` | `X0[t] = E_tok[id_t] + E_pos[t]` |
| `LN(x)` | `[d]` | `γ ⊙ (x − μ)/sqrt(σ² + ε) + β`, `ε = 1e-5`, Varianz mit `1/d` |
| `MHA` | `[T, d] → [T, d]` | Kapitel 05, `headDim = h · dK = d` |
| `W1`, `b1` | `[H, d]`, `[H]` | `u = GELU(W1 x + b1)` pro Zeile |
| `W2`, `b2` | `[d, H]`, `[d]` | `FFN(x) = W2 u + b2` |
| Block | `[T, d] → [T, d]` | `A = X + MHA(LN1(X))`, `Y = A + FFN(LN2(A))` |
| `W_out` | `[d, V]` | Output-Projektion, nicht an `E_tok` gekoppelt |
| Logits `Z` | `[T, V]` | `Z = LN_f(X_L) · W_out` |

GELU verwendet die tanh-Näherung
`0.5 x (1 + tanh(sqrt(2/π) (x + 0.044715 x³)))`.

Parameterzahl (mit Default `useQkvBias = useOutputBias = false`):

```text
P = V·d + T·d + L·(4d + 4d² + 2dH + H + d) + 2d + d·V
```

## Kapitelstruktur

1. Überblick decoder-only GPT mit Mermaid-Diagramm.
2. `GPTConfig` und Validierungsregeln.
3. Eingabe: Token- plus Positions-Embeddings.
4. Pre-LN-Transformer-Block: LayerNorm, Attention-Verweis, FFN/GELU, Residuals.
5. Finale LayerNorm und Vokabular-Logits.
6. Vollständige Shape-Tabelle und Parameterzahl mit Beispiel.
7. Kausalität auf Modellebene.
8. Tests, Design-Entscheidungen, Stolpersteine, Zusammenfassung, Überleitung zu Kapitel 07.

## Abgrenzung

- Kein Training, kein Optimierer, keine Generierung (→ Kapitel 07).
- Keine neue Attention-, Autograd- oder Modellklasse.
- Keine Herleitung der Attention-Mathematik über einen Verweis auf Kapitel 05 hinaus.
- Kein Weight-Tying, KV-Cache, RoPE oder Post-LN als implementierter Pfad.

## Befund und minimaler Fix

`LayerNormMultikTensor.forward` erzeugte einen `TensorMultik` mit Kindern, aber
ohne `backwardStep`. Dadurch endete jeder Gradient an einer LayerNorm: Token-
und Positions-Embeddings, alle Block-Parameter und `γ/β` erhielten keinen
Gradienten; nur `W_out` lernte. Nachweis: neuer Test
`GPTModelMultikTensorTest` schlug vor dem Fix für Parameter 0 fehl.
Fix: `forward` delegiert an die bereits kanonische Operation
`TensorMultik.layerNorm` (keine duplizierte Mathematik mehr). Abgesichert durch
`LayerNormMultikTensorTest` (Finite-Differenzen für Eingabe, `γ`, `β`).

## Messbare Akzeptanzkriterien

- [x] Kapitel `06-gpt.adoc` existiert, Anker `[[GPT]]`, in `book.adoc` nach Kapitel 05 eingebunden.
- [x] Formeln für Embedding-Summe, LayerNorm, FFN/GELU, Pre-LN-Residuals und Logits entsprechen dem Code.
- [x] Shape-Tabelle deckt `tokenIds` bis `Z = [T, V]` ab.
- [x] Parameterzahl-Formel ist durch einen Test belegt.
- [x] Kausalität auf Modellebene ist durch einen Test belegt.
- [x] Jeder Modellparameter erhält im Backward-Pfad einen Gradienten (Test).
- [x] Code wird über Include-Tags aus den Originaldateien eingebunden.
- [x] Verweise auf Kapitel 05 (<<Attention>>) und Kapitel 07 (<<Training>>).

## Quellen und Copyright

Primärquellen sind die Kotlin-Dateien und Tests im Repository. Standardnotation
(Pre-LN-Transformer, GELU-Näherung) wird eigenständig formuliert; es werden
keine fremden Texte, Abbildungen oder Codeblöcke übernommen. Die Architektur
orientiert sich laut `README.md` an Raschkas Buch; darauf wird nur verwiesen.

## Risiken und offene Punkte

- Behoben: Modellduplikat `kapitel4/GPTModelMultikTensor.kt` entfernt; Scratch-`GPTModel`, `GPTConfig`, `GenerationConfig` und die GPT-Demos liegen jetzt ausschliesslich in `kapitel5` (`model`, `demo`).
- ~~`ReStrukturierung.md` fordert Training über `TrainableTokenEmbedding`; das Modell besitzt eigene Embedding-Tensoren.~~ **Entschieden (Variante A):** `Trainable*` bleibt didaktisch (Kapitel 03, manueller Backward + SGD); das GPT-Modell nutzt die Autograd-Klasse `kapitel4/library/autograd/EmbeddingMultikTensor` für Token- und Positions-Embedding. `ReStrukturierung.md` ist entsprechend angepasst.
