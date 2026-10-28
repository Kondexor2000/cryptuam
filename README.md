# AI Research & Development Lab

Repozytorium zawiera prototyp chatbota oraz generator propozycji eksperymentu dotyczącego nauki słownictwa. Generator korzysta z lokalnych list w `docs_listen` i `docs_speech` oraz opcjonalnych wykluczeń w `docs_not`; nie trenuje modelu ani nie przedstawia wyników empirycznych.

## Generator briefu badawczego

Uruchom z katalogu projektu:

```bash
python research_lab.py
```

Aby otrzymać wynik maszynowy:

```bash
python research_lab.py --json
```

Domyślnie skrypt:

- czyta niepuste wiersze z `docs_listen/*.txt` i `docs_speech/*.txt` jako materiał do nauki;
- traktuje niepuste wiersze z `docs_not/*.txt` jako wpisy do wykluczenia;
- wyklucza tylko dokładne dopasowania po normalizacji wielkości liter, odstępów i interpunkcji;
- raportuje liczbę wpisów źródłowych, uwzględnionych i wykluczonych oraz przykłady i niedopasowane wykluczenia.

Pliki Markdown i inne pliki niż `.txt` nie są wczytywane jako listy słownictwa. Katalogi można wskazać jawnie; powtórz `--docs-dir`, aby połączyć kilka źródeł:

```bash
python research_lab.py --docs-dir ścieżka/do/docs_listen --docs-dir ścieżka/do/docs_speech --exclusions-dir ścieżka/do/docs_not
```

Założenia pytania badawczego i proponowany eksperyment opisuje [brief badawczy](docs_speech/RD_RESEARCH_BRIEF.md).

## Testy

Testy generatora sprawdzają wczytywanie list, dokładne wykluczanie wpisów, obsługę niepasujących fraz oraz format JSON. Uruchom je poleceniem:

```bash
python -m unittest research_lab_tests.py
```

## Chatbot

Uruchom `python chat.py`, aby użyć materiałów z `docs_speech` do obsługi wpisanego tekstu. Pomysły pojawiają się już na starcie. Chatbot pokazuje osobno pomysły z biblioteki oraz pomysły współpracujące z listami z `docs_listen`; po wpisaniu tekstu drugi zestaw uwzględnia także pytanie użytkownika. Każdy zestaw zawiera wariant zewnętrzny i wewnętrzny, po polsku i angielsku.

`index.py` buduje indeks z `docs_speech`. Używa embeddingów TensorFlow Hub, a przy błędzie ładowania modelu przechodzi na TF-IDF. `research_lab.py` jest odrębnym narzędziem i nie korzysta z indeksu chatbota.
