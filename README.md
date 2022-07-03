### PROJEKT (własny) z Politechniki Gdańskiej

Tytuł pracy w języku polskim:  <b> Analiza danych filmowych na platformie Netflix </b>

Tytuł pracy w języku angielskim:  <b> Analysis of movie data on the Netflix platform </b>

#### Wstęp
Projekt dotyczy analizy danych filmowych na platformie Netflix.
Projekt oparty jest na wybranych danych  dotyczących Systemu Rekomendacji Filmów.
- tytuły filmów
- czas wydania, najważniejsze gatunki filmów
- oceny i daty jej wystawienia

#### Cel projektu 
1. Nie jest związany z wszystkimi pobranymi danymi.
2. Nie wykorzystuję indywidualnych danych związanych z użytkownikami.
3. Nie chodzi o rekomendowanie, reklamowanie tytułów widzom na podstawie zebranych o nich danych, prefencji.

Najpierw przeprowadziłem prognozowanie wyników w przedziale czasu obejmującym zebrane dane (interpolacja).
Potem prognozowałem przyszłe wyniki (do 2030 r.) na podstawie zebranych danych (ekstrapolacja).

##### Badania oparłem na regresji wielomianowej.
- Dla znajdowania wartości pośrednich w obecnych czasach wykorzystałem metodę interpolacji.
- Prognozowanie wyników w przyszłości oparłem na ekstrapolacji. 

```
Projekt oparty został na otwartych danych z platformy Kaggle.
AUTOR zbioru zebranych danych (z Kaggle): Bandi Karthik
Movie Recommendation System
```

