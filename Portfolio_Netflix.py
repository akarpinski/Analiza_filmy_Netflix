#!/usr/bin/env python
# coding: utf-8

# ### Studia podyplomowe: Inżynieria Danych – Data Science
# PROJEKT (własny) z Politechniki Gdańskiej
# #### <i>autor: Artur Karpiński</i>

# ### Analiza danych filmowych na platformie Netflix
# Mój projekt oparty jest na wybranych danych  dotyczących Systemu Rekomendacji Filmów. Nie wykorzystuję indywidualnych danych związanych z użytkownikami. W odróżnieniu od autora danych nie chodzi mi więc o rekomendowanie, reklamowanie tytułów widzom na podstawie zebranych o nich danych, prefencji.
# Wybrane dane:
# - tytuły filmów
# - czas wydania, najważniejsze gatunki filmów
# - oceny i daty jej wystawienia
# 
# Na podstawie zebranych danych szukałem, analizowałem i pokazywałem zależności  między tytułami, gatunkami filmów, ich ocenami i popularnością wśród widzów na przestrzeni czasu. Korzystałem z otwartych danych z platformy Kaggle.

# ### Czyszczenie danych
# Wykorzystywać będziemy dane dwóch plików:
# 1. movies – informacje dotyczące filmów
# 2. ratings – informacje dotyczące ocen tych filmów

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


# informacje o filmach

df_movies = pd.read_csv('./data/movies.csv')


# In[3]:


# tabela zawiera więcej informacji niż ma kolumn
# w kolumnach więcej niż jedna informacja (np. rok, tytuł)

df_movies.head()


# In[4]:


# aby wyodrębnić tytuł i rok produkcji, string jako wzorzec wyrażenia regularnego poprzedzony r
# środku cztery liczby, expand=True, aby wzorzec z list string podzielić na kolumny
# po podziale tytułu na kolumny, usunięcie zbędnej, zmiana nazw kolumn

df_movies['title'].str.split(r'(\(\d\d\d\d\))', expand=True)     .drop(columns=2)     .rename(columns={0: 'title', 1: 'year'})     .head()


# In[5]:


# _zmienna pomocnicza _df i usunięcie nawiasów przy year

_df = df_movies['title'].str.split(r'(\(\d\d\d\d\))', expand=True)     .drop(columns=2)     .rename(columns={0: 'title', 1: 'year'})
df_movies['title'] = _df['title']
df_movies['year'] = _df['year'].str.strip('()')
del _df
df_movies.head()


# In[6]:


# lista i liczba brakujących elementów w year (68)

df_movies[df_movies['year'].isna()]


# In[7]:


# null nie jest konwertowalny do liczby
# brakujące wartości zastępujemy -1, bo taki rok na pewno nie istnieje

df_movies['year'] = df_movies['year'].fillna(-1).astype(np.int16)
df_movies['year'].dtype


# In[8]:


# lista i liczba brakujących elementów w year (teraz puste wyeliminowane)

df_movies[df_movies['year'].isna()]


# In[9]:


df_movies.head()


# In[10]:


# genres (rodzaje filmów) w jednej kolumnie trudne wykorzystania (analizy, filtrowania itp.)
# 'no genres listed' (brak rodzaju filmu) zamieniamy na pusty string bez kreski
# każdy wpis do małej litery
# dzielimy wpisy względem pionowej kreski

df_movies['genres'] = df_movies['genres'].replace({'(no genres listed)': ''})     .str.lower()     .str.split('|')
df_movies


# In[11]:


# teraz wyczyszczenie tabeli rating (oceny)
# wybieramy jedynie potrzebne kolumny

df_ratings = pd.read_csv('./data/ratings.csv', usecols=['movieId', 'rating', 'timestamp'])

df_ratings.head()


# In[12]:


# konwertujemy te liczby do obiektów typu datetime
# wykorzystujemy funkcję pd.to_datetime (sekundy jako jednostki)

df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')
df_ratings.head()


# In[13]:


# sprawdzamy czy timestamp zawiera jakieś informacje czasowe
# to liczba sekund od jakiegoś umownego roku

df_ratings['timestamp']


# In[14]:


# ocena najniższa

df_ratings['rating'].min()


# In[15]:


# ocena najniższa

df_ratings['rating'].max()


# ### Analiza danych
# #### Filmy z platformy Netflix.

# In[16]:


# oczyszczone dane o filmach

df_movies.head()


# In[17]:


# liczba filmów w poszczególnych latach
# loc[0:] filtruje filmy bez przypisanego roku (-1)

df_movies[['movieId', 'year']].groupby('year').count()


# In[18]:


# liczba filmów w poszczególnych latach
# nie do końca zdefiniowany koniec (nie wszystkie dane z ostatniego roku)

df_movies[['movieId', 'year']].groupby('year').count().loc[0:2014].plot(figsize=(12, 3))
plt.title('Liczba wszystkich filmów do roku 2014')
df_movies[['movieId', 'year']].groupby('year').count().loc[2000:2014].plot(figsize=(12, 3))
plt.title('Liczba filmów w latach 2000 – 2014')


# #### Cel: badanie związków między latami produkcji, gatunkami filmów, ich oceną

# In[19]:


# za pomocą funkcji lambda konwersja do listy każdego rzędu gatunków filmów 

df_movies['genres'].apply(lambda x: list(x))


# In[20]:


# dzięki explode na kolumnie genres gatunki ułożą się jeden pod drugim

df_movies['genres'] = df_movies['genres'].apply(lambda x: list(x))
df1 = df_movies.explode('genres')
df1


# In[21]:


# ilość gatunków filmów w każdym zbiorze

df1['genres'].value_counts()


# In[22]:


# wykres horyzontalny (kind='barth') od wartości najmniejszej

df1['genres'].value_counts().sort_values(ascending=True).plot(kind='barh', figsize=(12, 5))
plt.title('Ilość filmów w gatunkach')


# In[23]:


# często filmy są przydzielone do wielu kategorii
# dzięki 'value_counts' ilość tych wystąpień

df_movies['genres'].apply(len).value_counts()


# In[24]:


# wykres słupkowy (kind='bar')

df_movies['genres'].apply(len).value_counts().plot(kind='bar', figsize=(12, 3))
plt.title('Liczba gatunków przydzielonych filmom')


# In[25]:


# filmy po 1940 roku

df2 = df1[df1['year'] > 1940]
df2


# In[26]:


# tworzenie podgrup dla filmów w poszczególnych dekadach

# dla grupowania filmów po przedziałach w poszczególnych latach funkcja 'cut'
# 'bins' to kieszenie,'range' to zakres
# przedział dla lat 1940-2020 co 10 lat

pd.cut(df2['year'], bins=range(1940, 2030, 10), labels=[f"{x}s"     for x in range(40, 100, 10)] + ['2000s', '2010s'])


# In[27]:


# liczba filmów w czasach, dekadach

df2['times'] = pd.cut(df2['year'], bins=range(1940, 2030, 10),     labels=[f"{x}s" for x in range(40, 100, 10)] + ['2000s', '2010s'])
df2


# In[28]:


# jak zmieniała się (wzrastała) liczba filmów w poszczególnych gatunkach

df2[['movieId', 'genres', 'times']]


# In[29]:


# grupowanie po parach gatunek-czas zliczając 'movieId'

# zmiana nazwy 'movieId' na 'decades'
# gdy gdzieś none to zastępujemy 0, zmieniamy typ na int
# unstack pozwala rozerwać długi DataFrame i pogrupować indeksy jeden obok drugiego
# jednym indeksem jest 'genres' filmu, a drugim 'decades'

df3 = df2[['movieId', 'genres', 'times']]     .groupby(by=['genres', 'times'])     .count()     .rename(columns={'movieId': 'decades'})     .fillna(0)     .astype(int)     .unstack()
df3


# In[30]:


# rozwój gatunków na przestrzeni czasu

# heatmap dla pokazania trendów produkcji w poszczególnych latach
# płótno wykresu przez 'subplots', mapa kolorów przez 'cmap'
# 'annot=True' dla wyświetleń liczbowych, 'fmt' dla usunięcia ułamków
# 'linewidth' dla rozdzielenia linii

fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(df3, cmap='hot', annot=True, fmt='1', linewidth=0.5)
plt.title('Popularność gatunków w czasie')


# #### Cel: badanie ocen poszczególnych filmów
# Potem wnioski z korelacji (zależności) między filmami, a ich ocenami

# In[31]:


df_ratings.head()


# In[32]:


# cel: badanie związków między latami produkcji, gatunkami filmów, ich oceną

df_ratings.info()


# In[33]:


# widzowie oceniali dodając ilość punktów, gwiazdek
# oceny między 0.5 oraz 5.0

df_ratings['rating'].min()


# In[34]:


df_ratings['rating'].max()


# In[35]:


# rozkład nie jest równomierny, a średnia między 3 i 4

df_ratings['rating'].plot(kind='hist', bins=10, width=0.3)
plt.title('Liczba ocen filmów')


# In[36]:


# ilość wystąpień ocen

df_ratings['rating'].value_counts()


# In[37]:


# łączymy obie tabele ('movies','ratings') po 'movieId'

# w tabeli 'ratings' liczymy średnią, odchylenie standardowe,  liczbę ocen
# wybieramy tylko filmy z ponad 100 głosami widzów
# 'dfq' zawiera oceny

dfq = df_ratings[['movieId', 'rating']]     .groupby('movieId')     .agg({'rating': ['mean', 'std', 'count']})
dfq = dfq[dfq['rating']['count'] > 100]
dfq


# In[38]:


# po przycięciu tabeli względem ilości widzów dalsze uproszczenie

# uproszczenie przez wybór tabeli z wartością średnią i odchyleniem standardowym
# funkcja 'concat' łączy obiekty pandy wzdłuż określonej osi

dfq = pd.concat([dfq['rating']['mean'], dfq['rating']['std']], axis=1).reset_index()
dfq


# In[39]:


# histogram histplot z biblioteki seaborn
# rozkład (przegląd) średnich ocen dla wszystkich filmów

sns.histplot(dfq['mean'], bins=np.linspace(0, 5, num=26))
plt.title('Rozkład średniej ocen filmów')


# In[40]:


# najgorsze filmy z lewej strony wykresu

# dla nich wartości mniejsze niż w skrajnym kwantylu (pół procent)
# ze związku między mean i std widać, że oceny skrajnie negatywne

bad_movies = dfq[dfq['mean'] < dfq['mean'].quantile(0.005)]
bad_movies.head()


# In[41]:


# dla najlepszych filmów analogicznie

# zmiana nierówności i skrajmych kwantyli
# najlepsze filmy były z prawej strony wykresu

good_movies = dfq[dfq['mean'] > dfq['mean'].quantile(0.995)]
good_movies.head()


# In[42]:


# średnia ocen

dfq['mean'].mean()


# #### Najgorsze filmy

# In[43]:


# poniżej lista najgorzej ocenianych filmów

# sortowanie względem średniej i w sposób rosnący
# przez 'merge' dobieramy (scalamy) do 'bad_movies' na indeksie 'movieId'

# najgorszym był "Battlefield Earth"

dfm = df_movies[['movieId', 'title', 'genres', 'year']]

dfm.merge(bad_movies, on='movieId')     .sort_values(by='mean', ascending=True)     .head()


# #### Najlepsze filmy

# In[44]:


# poniżej lista najwyżej ocenianych filmów

# sortowanie względem średniej i w sposób malejący

# najlepszym był "The Shawshank Redemption"

dfm.merge(good_movies, on='movieId')     .sort_values(by='mean', ascending=False)     .head()


# #### Najbardziej popularne filmy
# 

# In[45]:


# lista głosów najpopularniejszych filmów

df_ratings[['movieId', 'rating']]     .groupby('movieId')     .count()     .sort_values(by='rating', ascending=False)     .rename(columns={'rating': '#votes'})     .head()


# In[46]:


# za pomocą 'merge' można dodać 'dfm' (plik tytułów filmów)

# najpopularniejszym był "Forrest Gump"

df_ratings[['movieId', 'rating']]     .groupby('movieId')     .count()     .sort_values(by='rating', ascending=False)     .rename(columns={'rating': '#votes'})     .merge(dfm, left_index=True, right_on='movieId', how='left')     .head()


# #### Ocena a popularność

# In[47]:


# nie były to najwyżej oceniane filmy lecz z największą liczbą głosów
# nie wiemy czy najlepiej oceniane te najbardziej popularne

# do 'dfm' (tytuły) za pomocą 'merge' można też dodać 'dfq' (plik z ocenami)
# sortowanie przez 'std' daje pojęcie o spójności ocen

# czyli jednym się podobało, innym nie, ale ogólnie dobre oceny, popularność

dfc = df_ratings[['movieId', 'rating']]     .groupby('movieId')     .count()     .sort_values(by='rating', ascending=False)     .rename(columns={'rating': '#votes'})     .merge(dfm, left_index=True, right_on='movieId', how='left')     .merge(dfq, on='movieId', how='inner')     .sort_values(by='std', ascending=False)
dfc


# In[48]:


# średnia ocen

dfq['mean'].mean()


# #### Porównanie 2 najpopularniejszych gatunków

# In[49]:


# kopiowanie wartości do nowego DataFrame, aby wartości się nie nadpisywały
# przez apply zmieniamy zbiory na listy
# eksplodowanie tych gatunków, by ustawiły się jako rzędy jeden pod drugim

dfg = dfc[['#votes', 'mean', 'std', 'genres']].copy()
dfg['genres'] = dfg['genres'].apply(lambda x: list(x))
dfg = dfg.explode('genres')
dfg


# In[50]:


# porównanie 2 najpopularniejszych gatunków: 'drama', 'comedy'

dfg[dfg['genres'].isin(['drama', 'comedy'])]


# In[51]:


# można pokazać na 'pairplot'
# parametr 'hue' podzeli ten wykres na 2 podwykresy względem gatunku

# komedie i dramaty tak samo popularne
# jednak ocena dramatu statystycznie wyższa niż komedii
# oceniający też bardziej spójni przy dramatach (mniejsze std) niż komediach
# wg widzów lepiej obejrzeć dobry dramat niż średnią komedię

sns.pairplot(dfg[dfg['genres'].isin(['drama', 'comedy'])], hue='genres')


# #### Wnioski:
# 1. czyli niekoniecznie wysoka popularność jest zgodna z oceną
# 2. widzowie mogą też zwracać uwagę na różne elementy
# 3. np. widzowie mają ulubione gatunki filmów, a unikają innych

# In[52]:


# jak zmieniały się oceny filmów w miesięcznych przedziałach czasu

# rozkład głosów względem czasu za pomocą funkcji mean
# średnia między 3, a 4

df_ratings[['rating', 'timestamp']].resample('M', on='timestamp').mean().plot(figsize=(14, 4))
plt.title('Oceny filmów w czasie')


# In[53]:


# jak zmieniała się  popularność platformy w przedziałach czasu

# 'count' zlicza głosy w tych przedziałach
# mamy film i czas, w którym była ocena
# funkcja 'resample' działa podobnie jak 'groupby'
# agreguje dane w czasie ('M' to miesiąc, 'timestamp' to funkcja agregowana)

df_ratings[['movieId', 'timestamp']].resample('M', on='timestamp').count().plot(figsize=(14, 4))
plt.title('Popularność platformy w czasie')


# #### "Star Wars" – "Gwiezdne Wojny"

# In[54]:


# ograniczenie się tylko do serii "Gwiezdne Wojny"

# 'Star Wars: Episode' ogranicza się do 7 filmów
# str.contains() sprawdza czy wyrażenie regularne jest zawarte w ciągu

dfsw = dfm[dfm['title'].str.contains('Star Wars: Episode')].sort_values(by='title')
dfsw


# In[55]:


# przez 'range' (przedział, zakres) numerowanie części 'Star Wars: Episode'
# tworzenie nowej kolumny 'episode' z numerami 7 części filmu
# odrzucenie kolumny 'genres'(gatunki), zmiana kolejności pozostałych 

dfsw = dfm[dfm['title'].str.contains('Star Wars: Episode')].sort_values(by='title')
dfsw['episode'] = range(1, 8)
dfsw[['movieId', 'episode', 'title', 'year']]


# In[56]:


# przez 'merge' łączenie dwóch DataFrame
# dodanie ocen tych filmów łącząc z tabelą 'df' na wspólnym kluczu 'movieId'
# mamy dane wszystkich części 'Star Wars' z ocenami i czasem ich wystawienia

dfsw = dfm[dfm['title'].str.contains('Star Wars: Episode')].sort_values(by='title')
dfsw['episode'] = range(1, 8)
dfsw = dfsw[['movieId', 'episode', 'title', 'year']].merge(df_ratings, on='movieId')
dfsw


# In[57]:


# stworzenie pustego słownika
# z tabeli będziemy wybierać mniejsze tabele odwołując się do konkretnej części filmu
# z otrzymanej tabeli 3 kolumny ('rating', 'year', 'timestamp')
# na tej tabeli wywołamy funkcję 'resample', gdzie będziemy zliczać średnie w przedziałach rocznych
# 'Y' jako pierwszy argument, 'mean' jako funkcja agregująca
# wynik zapisany do słownika 'sw[i]'

sw = {}
for i in range(1, 8):
    sw[i] = dfsw[dfsw['episode'] == i][['rating', 'year', 'timestamp']]         .resample('Y', on='timestamp').mean()


# In[58]:


# mamy więc tabelę, w której kluczem będzie 'timestamp'
# następnie będzie 'rating' i rok produkcji 'year'

sw[1]


# In[59]:


# następnie zresetowanie tego indeksu i stworzenie nowej kolumny
# stworzenie kolumny 'year_since' (ile lat upłynęło od ukazania się filmu)
# wybierzemy rok i odejmujemy rok produkcji

sw = {}
for i in range(1, 8):
    sw[i] = dfsw[dfsw['episode'] == i][['rating', 'year', 'timestamp']]             .resample('Y', on='timestamp').mean()
    sw[i] = sw[i].reset_index()
    sw[i]['years_since'] = sw[i]['timestamp'].dt.year - sw[i]['year']


# In[60]:


sw[1]


# In[61]:


# kolumna 'year' już nieistotna
# indeksem nowej tabeli będzie 'years_since'
# otrzymamy tabelą z indeksem, którym jest ilość lat od produkcji
# drugą kolumną 'rating' (ocena średnia w tym roku)

sw = {}
for i in range(1, 8):
    sw[i] = dfsw[dfsw['episode'] == i][['rating', 'year', 'timestamp']]             .resample('Y', on='timestamp').mean()
    sw[i] = sw[i].reset_index()
    sw[i]['years_since'] = sw[i]['timestamp'].dt.year - sw[i]['year']
    sw[i] = sw[i][['rating', 'years_since']].set_index('years_since')


# In[62]:


sw[1]


# In[63]:


# takie tabele możemy złączyć w jedną używając funkcji 'concat'
# uzyskujemy podobny wynik dla każdej części filmu "Star Wars"
# dla ostatniej kolumny tylko dwie wartości, bo film ukazał się najpóźniej
# część 4,5,6 powstały najwcześniej

pd.concat(sw, axis=1)


# In[64]:


# wyświetlenie wyników przez 'concat' i dodatkowo funkcję 'plot'

# nowsze części filmu oceniane gorzej niż wcześniejsze
# z upływem czasu widać też spadek ocen

# te wcześniejsze części filmu dużo lepiej oceniane
# również dużo bardziej stałe w swojej ocenie
 
pd.concat(sw, axis=1).plot(figsize=(14, 5))
plt.title('Oceny poszczególnych części filmu w czasie')


# In[65]:


# części 4,5,6 powstały najwcześniej
# prawdopodobnie starsze wersje wyżej oceniane niż wcześniejsze

dfsw = dfm[dfm['title'].str.contains('Star Wars: Episode')]
dfsw = dfsw[['movieId', 'title', 'year']].merge(df_ratings, on='movieId').sort_values(by='rating', ascending=False)
dfsw.head()


# #### Wnioski
# 1. nowsze części filmu oceniane gorzej niż wcześniejsze
# 2. te wcześniejsze części filmu dużo lepiej oceniane
# 3. również dużo bardziej stałe w swojej ocenie
# 

# ### Zastosowanie regresji wielomianowej
# 
# #### 1 przykład – "Star Wars: Episode 1" w interpolacji
# Wykorzystanie do znajdowania wartości pośrednich w obecnych czasach bez prognozowania w przyszłości.

# In[66]:


# zmiana reprezenacji danych z pandasowego DataFame na tablicę w Numpy
# najpierw 'reset_index' i podajemy przedział indeksów w 'numpy'
# 'x' będzie liczbą lat od produkcji, a 'y' będzie oceną 
# wykorzystując te 'x','y' chcemy zastosować regresję wielomianową

x = sw[1].reset_index().to_numpy()[:, 0]
y = sw[1].reset_index().to_numpy()[:, 1] 
y


# In[67]:


# współczynniki wielomianu uzyskujemy przez funkcję 'polyfit'
# robimy to na 'x','y' przekazując 'deg' (stopień wielomianu)
# 't' podnosimy do każdej kolejnej potęgi, stąd 'deg+1'
# będą więc 4 współczynniki zaczynając od 0

# zastosować można poniżej przedstawiony algorytm

from numpy import polyfit

# 'linspace' zwraca liczby w określonym przedziale
# polyfit - dopasowanie wielomianowe metodą najmniejszych kwadratów
# dopasowuje wielomian stopnia 'deg' do (x,y)
# concatenate - łączy sekwencję wzdłuż istniejącej osi
# sprawdzimy optymalny stopień wielomianu (np. 2,3,4)

t = np.linspace(0, 20, num=21)

#deg = 2
deg = 3
#deg = 4

coeffs = polyfit(x, y, deg=deg).reshape(-1, 1)
X = np.concatenate(list(map(lambda i: t.reshape(1, -1) ** i, reversed(range(deg + 1)))))
Y = (coeffs.T @ X).flatten()


# In[68]:


# na wykresie oceny widzów dla 1 części "Star Wars"

# szkielet wykresu przy użyciu funkcji 'subplots'
# wywołamy 'ax.plot', potem sw[1] dla 1 części i oznaczymy dane przez kółko

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(sw[1], 'o')
plt.title('Przedziały ocen od kilkunastu lat')


# In[69]:


# następnie nasze przewidywania na podstawie regresji wielomianowej

# widzimy, że wielomian 3-go stopnia był dobrym dopasowaniem
# gdy zmienimy na 2-go stopnia, to nie mamy dobrego dopasowania
# dla 4-go stopnia dopasowanie staje się tak dobre, że mało realistyczne
# nie dałoby się przewidzieć co stałoby się z głosami widzów później

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(sw[1], 'o')
ax.plot(t, Y)
plt.title('Przewidywane wyniki między znanymi węzłami danych')


# #### 2 przykład – wszystkie dane w ekstrapolacji
# Prognozowanie wartości zmiennej lub funkcji poza zakresem, dla którego mamy dane.
# 
# Dopasowanie do istniejących danych pewnej funkcji, następnie wyliczenie jej wartości w szukanym punkcie w przyszłości.

# In[70]:


# ilość filmów w poszczególnych latach

dfm[['movieId', 'year']].groupby('year').count()


# In[71]:


# tendencje w formie wykresu

dfm[['movieId', 'year']].groupby('year').count().loc[0:2014].plot(figsize=(12, 3))
plt.title('Ilość filmów w poszczególnych latach')


# In[72]:


# najpierw zamienimy tę tabelę na 'numpy'
# można teraz na tych danych dokonać regresji wielomianowej
# naszymi 'x' będzie rok produkcji, 'y' liczba filmów w konkretnych latach

X = dfm[['movieId', 'year']].groupby('year').count().loc[0:].reset_index().to_numpy()


# In[73]:


X


# In[74]:


# 'X' to ilość filmów do roku 2014
# dane ograniczone do roku 2014, by uniknąć wyników niewiarygodnych

X = dfm[['movieId', 'year']].groupby('year').count().loc[0:2014].reset_index().to_numpy()


# ##### Co stanie się z filmami po 2030 roku?
# W ekstrapolacji stosujemy algorytm podobny do interpolacji w „Star Wars: Episode 1”.
# Zmiana odpowiednich parametrów, wartości zmiennych.
# 
# Na wykresach przewidywane wyniki (linia) między znanymi węzłami danych (kropki).
# Sprawdzę kolejno, czy optymalny wielomian będzie stopnia 2,3,4.

# In[75]:


# 't' będzie znowu zmienną, dzięki której będziemy przewidywać
# np. co stanie się z filmami w 2030 r.
# kopiujemy kod, który już mamy (z 1 przykładu)
# zamieniamy tylko 'x','y' na 'X'
# zerowy indeks to rok produkcji, a pierwszy indeks to ilość filmów
# 'Y' to ilość filmów wyświetlanych w konkretnych latach

# następnie sprawdźmy kolejno, czy optymalny wielomian będzie stopnia 2,3,4

deg = 2

X = dfm[['movieId', 'year']].groupby('year').count().loc[0:2014].reset_index().to_numpy()

t = np.linspace(1900, 2030, num=41)

coeffs = polyfit(X[:, 0], X[:, 1], deg=deg).reshape(-1, 1)
X = np.concatenate(list(map(lambda i: t.reshape(1, -1) ** i, reversed(range(deg + 1)))))
Y = (coeffs.T @ X).flatten()


# In[76]:


# dla wielomianu 2-go stopnia nie mamy dobrego dopasowania (zbyt odstający)

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(dfm[['movieId', 'year']].groupby('year').count().loc[0:2014], 'o')
plt.title('Ilość filmów do roku 2014')

# dodajemy nasz wielomian (czyli 't' oraz 'Y')
# tworzymy wykres oryginalnych danych (kropki)
# przewidywania czerwoną linią

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(dfm[['movieId', 'year']].groupby('year').count().loc[0:2014], 'o')
ax.plot(t, Y, 'r')
plt.title('Przewidywana ilość filmów do roku 2030          (wielomian 2 stopnia)')


# In[77]:


# wielomian 3-go stopnia wydaje się dobry

# mógłby symulować, w jaki sposób ilość fimów wzrastałaby w przyszłości

deg = 3

X = dfm[['movieId', 'year']].groupby('year').count().loc[0:2014].reset_index().to_numpy()

coeffs = polyfit(X[:, 0], X[:, 1], deg=deg).reshape(-1, 1)
X = np.concatenate(list(map(lambda i: t.reshape(1, -1) ** i, reversed(range(deg + 1)))))
Y = (coeffs.T @ X).flatten()

# dodajemy nasz wielomian (czyli 't' oraz 'Y')
# przewidywania zaznaczone zieloną linią

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(dfm[['movieId', 'year']].groupby('year').count().loc[0:2014], 'o')
ax.plot(t, Y, 'g')
#ax.plot(t, Y, 'r')
plt.title('Przewidywana ilość filmów do roku 2030          (wielomian 3 stopnia)')


# In[78]:


# dla 4-go stopnia dopasowanie staje się tak dobre, że mało realistyczne (model nie uczy się)

# przy stopniu wyższym od 3 nie dałoby się przewidzieć co stałoby się z głosami widzów później
# sprawdzałem też dla wyższych niż pokazany niżej na wykresie 4-go stopnia

deg = 4

X = dfm[['movieId', 'year']].groupby('year').count().loc[0:2014].reset_index().to_numpy()

coeffs = polyfit(X[:, 0], X[:, 1], deg=deg).reshape(-1, 1)
X = np.concatenate(list(map(lambda i: t.reshape(1, -1) ** i, reversed(range(deg + 1)))))
Y = (coeffs.T @ X).flatten()

# dodajemy nasz wielomian (czyli 't' oraz 'Y')
# przewidywania zaznaczone czerwoną linią

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.plot(dfm[['movieId', 'year']].groupby('year').count().loc[0:2014], 'o')
ax.plot(t, Y, 'r')
ax.plot(t, Y, 'r')
plt.title('Przewidywana ilość filmów do roku 2030          (wielomian 4 stopnia)')


# #### Porównanie wyników na podstawie regresji wielomianowej
# 
# 1. Widzimy, że wielomian 3-go stopnia był dobrym dopasowaniem.
# 
# 2. Gdy zmienimy na 2-go stopnia, to nie mamy dobrego dopasowania (zbyt odstający).
# 
# 3. Dla 4-go stopnia (i wyższych) dopasowanie staje się tak dobre, że mało realistyczne.
# 
# 4. Przy stopniu wyższym od 3 nie dałoby się przewidzieć co stałoby się z głosami widzów później.
# 
# #### Mieliśmy 2 przykłady wykorzystania regresji wielomianowej
# Pierwszy - w <u>interpolacji</u>, drugi - w <u>ekstrapolacji</u>.
# 1. Pierwszy - przy "Star Wars" oceny można było wykorzystać do znajdowania wartości pośrednich w obecnych czasach (interpolacja).
# 2. Drugi - przy ilości wszystkich produkowanych filmów, do przewidywania szybkości wzrostu lub spadku w przyszłości (ekstrapolacja).
