<img src="Portfolio_Netflix/PG_WFTiMS.jpg" alt="Politechnika Gdańska" width="300" height="auto"> 

##### PROJEKT (własny) na studiach "Inżynieria danych - Data Science" na Politechnice Gdańskiej.

- kod w języku Python na edytorze Jupyter Notebook
- wizualny podgląd w przeglądarce z pliku w Python (ipynb) lub HTML
- dokładna, pełna dokumentacja projeku w pliku PDF

---

### Analiza danych filmowych na platformie Netflix

#### Wstęp
Projekt dotyczy analizy danych filmowych na platformie Netflix.

Oparty jest na wybranych danych  dotyczących Systemu Rekomendacji Filmów.
- tytuły filmów
- czas wydania, najważniejsze gatunki filmów
- oceny i daty jej wystawienia

#### Cel projektu 
1. Nie jest związany z wszystkimi pobranymi danymi.
2. Nie wykorzystuję indywidualnych danych związanych z użytkownikami.
3. Nie chodzi o rekomendowanie, reklamowanie tytułów widzom na podstawie zebranych o nich danych, prefencji.

```
Najpierw prowadziłem prognozowanie wyników w przedziale czasu obejmującym zebrane dane (interpolacja).

Potem prognozowałem przyszłe wyniki (do 2030 r.) na podstawie zebranych danych (ekstrapolacja).
```

##### Badania oparłem na regresji wielomianowej.
- Interpolacja – metoda znajdowania wartości pośrednich w obecnych czasach obejmujących zebrane dane.
- Ekstrapolacja – metoda prognozowania przyszłych wyników na podstawie zebranych wcześniej danych. 

```
Projekt oparty został na otwartych danych z platformy Kaggle.
##### źródło analizowanych danych - https://www.kaggle.com
AUTOR zbioru zebranych danych (z Kaggle): Bandi Karthik
Movie Recommendation System
```

