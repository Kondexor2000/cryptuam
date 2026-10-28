# Brief badawczy: mieszana nauka słownictwa

**Status:** propozycja eksperymentu. Nie jest wynikiem badania ani potwierdzeniem skuteczności metody.

## Pytanie badawcze

Czy mieszanie słów z różnych list podczas ćwiczeń poprawia ich odroczone przypominanie w porównaniu z nauką listami, jeśli czas nauki i ekspozycja na każde hasło są takie same?

Generator pobiera materiał z niepustych wierszy plików `docs_listen/*.txt` i `docs_speech/*.txt`. Wiersze z `docs_not/*.txt` służą do odfiltrowania dokładnych dopasowań. Dopasowanie ignoruje wielkość liter, różnice w odstępach i interpunkcję, ale nie usuwa wpisu tylko dlatego, że zawiera wykluczone słowo jako część dłuższej frazy. Wygenerowany brief raportuje rzeczywistą liczbę i pochodzenie pozycji dla bieżącej zawartości katalogów.

## Hipoteza

Przy tej samej liczbie prezentacji i tym samym czasie nauki osoby ćwiczące na przemieszanej puli haseł będą osiągały wyższą skuteczność odroczonego przypominania niż osoby uczące się listami z poszczególnych plików.

## Proponowany eksperyment

1. Użyć kwalifikujących się haseł z `docs_listen/*.txt` i `docs_speech/*.txt`; przed badaniem zapisać ich wersję oraz listę wykluczeń z `docs_not/*.txt`.
2. Losowo przydzielić uczestników do warunku mieszanego lub do nauki listami odpowiadającymi plikom źródłowym.
3. Wyrównać łączny czas ćwiczeń, podpowiedzi i liczbę ekspozycji na każde hasło.
4. Przeprowadzić test przypominania bezpośrednio po nauce oraz test odroczony po ustalonym, jednakowym dla wszystkich czasie.
5. Powtórzyć badanie na wcześniej ustalonych grupach lub ziarnach losowych i raportować niepewność wyników.

## Miary

- poprawność przypominania bezpośredniego i odroczonego;
- skuteczność osobno dla grup haseł pochodzących z każdego pliku;
- czas nauki przypadający na poprawnie przypomniane hasło;
- liczba haseł włączonych, wykluczonych i nierozpoznanych na liście wykluczeń.

## Kryterium falsyfikacji

Hipotezę należy uznać za niepotwierdzoną, jeśli nauka mieszana nie poprawia odroczonego przypominania przy wyrównanej ekspozycji lub efekt nie powtarza się w kolejnych grupach.

## Ograniczenia

- Nazwy plików i same listy nie gwarantują, że hasła są równoważne pod względem trudności, znajomości ani kategorii semantycznej.
- Filtrowanie jest dokładne po normalizacji; celowo nie usuwa częściowych dopasowań ani podobnych wariantów haseł.
- Wynik generatora jest planem badania, a nie pomiarem skuteczności.
- Wnioski wymagają kontrolowanego badania z uczestnikami, ustalonym protokołem i odpowiednią analizą statystyczną.
