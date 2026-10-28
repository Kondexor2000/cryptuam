# AI Research & Development Lab

Eksperymentalny projekt łączący prototyp chatbota z generatorem briefu badawczego dotyczącego nauki słownictwa. Generator korzysta z lokalnych list w `docs_listen` i `docs_speech` oraz opcjonalnych wykluczeń w `docs_not`; nie trenuje modelu ani nie przedstawia wyników empirycznych.

## Strona projektu

Repozytorium zawiera stronę React/Vite publikowaną w GitHub Pages oraz API FastAPI uruchamiające Pythonowe narzędzia. API czyta listy z repozytorium bezpośrednio; nie używa ani nie wymaga bazy danych.

```bash
npm install
npm run dev
```

Build produkcyjny: `npm run build` (wynik w `dist/`). Workflow GitHub Actions publikuje stronę po pushu do gałęzi `main`. W ustawieniach repozytorium wybierz **Settings → Pages → Build and deployment → GitHub Actions**.

API można uruchomić lokalnie:

```bash
python -m pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

Endpointy: `GET /api/health`, `GET /api/brief` i `POST /api/chat` z JSON-em `{"question":"..."}`.

### Wdrożenie API za darmo

GitHub Pages obsługuje statyczny frontend, a darmowy Render uruchamia API Pythona. Utwórz usługę Render typu **Blueprint** dla repozytorium; gotowe ustawienia znajdują się w `render.yaml` i wybierają bezpłatny plan bez bazy danych. Render buduje i wdraża API po każdym pushu.

Przycisk wdrożenia: [Deploy to Render](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FKondexor2000%2Fcryptuam). URL API używany przez stronę to `https://cryptuam-python-api.onrender.com`; jeśli Render przydzieli inny adres, ustaw `VITE_API_URL` w kroku buildu Pages na ten adres.

Darmowe usługi Render usypiają po 15 minutach bez ruchu, więc pierwsze zapytanie po przerwie może czekać około minuty. API nie utrwala pytań ani odpowiedzi i nie wymaga bazy danych.

## Generator briefu badawczego

Uruchom z katalogu projektu:

```bash
python research_lab.py
```

Aby otrzymać wynik maszynowy:

```bash
python research_lab.py --json
```

Domyślnie skrypt czyta niepuste wiersze z `docs_listen/*.txt` i `docs_speech/*.txt` jako materiał do nauki, traktuje niepuste wiersze z `docs_not/*.txt` jako wykluczenia i usuwa tylko dokładne dopasowania po normalizacji wielkości liter, odstępów i interpunkcji. Raportuje liczbę wpisów źródłowych, uwzględnionych i wykluczonych oraz przykłady i niedopasowane wykluczenia.

Pliki Markdown i inne pliki niż `.txt` nie są wczytywane jako listy słownictwa. Katalogi można wskazać jawnie; powtórz `--docs-dir`, aby połączyć kilka źródeł:

```bash
python research_lab.py --docs-dir ścieżka/do/docs_listen --docs-dir ścieżka/do/docs_speech --exclusions-dir ścieżka/do/docs_not
```

Założenia pytania badawczego i proponowany eksperyment opisuje [brief badawczy](docs_speech/RD_RESEARCH_BRIEF.md).

## Testy generatora

```bash
python -m unittest research_lab_tests.py
```

## Chatbot

Uruchom `python chat.py`, aby użyć materiałów z `docs_speech` do obsługi wpisanego tekstu. `index.py` buduje indeks z `docs_speech`, używając embeddingów TensorFlow Hub lub awaryjnie TF-IDF. `research_lab.py` jest odrębnym narzędziem i nie korzysta z indeksu chatbota.
