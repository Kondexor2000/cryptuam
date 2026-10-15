# AI Research & Development Lab

Repozytorium zawiera prototypy chatbota opartego na dokumentach oraz mały generator propozycji eksperymentu dotyczącego nauki słownictwa. Generator korzysta z lokalnych list w `docs` i `docs_not`; nie trenuje modelu ani nie przedstawia wyników empirycznych.

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

- czyta niepuste wiersze z plików `docs/*.txt` jako materiał do nauki;
- traktuje niepuste wiersze z `docs_not/*.txt` jako wpisy do wykluczenia;
- wyklucza tylko dokładne dopasowania po normalizacji wielkości liter, odstępów i interpunkcji;
- raportuje liczbę wpisów źródłowych, uwzględnionych i wykluczonych oraz przykłady i niedopasowane wykluczenia.

Pliki Markdown i inne pliki niż `.txt` nie są wczytywane jako listy słownictwa. Katalogi można wskazać jawnie:

```bash
python research_lab.py --docs-dir ścieżka/do/docs --exclusions-dir ścieżka/do/docs_not
```

Założenia pytania badawczego i proponowany eksperyment opisuje [brief badawczy](docs/RD_RESEARCH_BRIEF.md).

## Testy

Testy generatora sprawdzają wczytywanie list, dokładne wykluczanie wpisów, obsługę niepasujących fraz oraz format JSON. Uruchom je poleceniem:

```bash
python -m unittest research_lab_tests.py
```

## Wcześniejszy prototyp chatbota

`index.py`, `chat.py` i `chat_tests.py` pochodzą z wcześniejszego prototypu chatbota bazującego na indeksie FAISS. `research_lab.py` nie wywołuje tych skryptów ani nie korzysta z indeksu. Są to odrębne ścieżki w repozytorium.
Chatbot wyświetla automatycznie wyodrębnione słowa kluczowe przy pomysłach zewnętrznych i wewnętrznych, po polsku i angielsku.
