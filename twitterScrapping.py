from bs4 import BeautifulSoup #importing the BeautifulSoup library
import requests # extracting data from websites
import csv # writing data to csv files
#from sklearn.feature_extraction.text import CountVectorizer # converting text to a matrix of token counts
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Tu clave de API y tu token de acceso
consumer_key = 'jaja'
access_token = 'jaja2'

# El nombre de usuario del que quieres recuperar los tweets
username = 'elonmusk'

# Envía la solicitud a la API de Twitter
response = requests.get(f'https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name={username}&count=10',
                        auth=(consumer_key, access_token))
print(response)
# Comprueba que la solicitud se haya realizado correctamente
if response.status_code == 200:
    # Procesa el contenido de la respuesta
    tweets = response.json()
    for tweet in tweets:
        print(tweet['text'])
        tweets.append(tweet['text'])

    # Los tweets que quieres clasificar
    print(tweets)
    # tweets = ['Este es un tweet sobre noticias', 'Este es un tweet sobre deportes', 'Este es un tweet sobre entretenimiento']

    # Estamos asignando una categoria negativo o positivo(0 o 1) a cada tweet en función si contiene palabras positivas o negativas.
    # Luego, el clasificador K-NN intentará predecir la categoría de un tweet que no haya visto antes. 

   # Abre el fichero con las palabras positivas
    with open('palabras_positivas.txt', 'r') as f:
        # Lee las palabras positivas del fichero
        positive_words = f.read().splitlines()

    # Asigna una categoría a cada tweet
    y = np.array([1 if any(word in tweet.lower() for word in positive_words) else 0 for tweet in tweets])
    
    # Crea el vectorizador de características
    vectorizer = CountVectorizer()

    # Crea una matriz de características a partir de los tweets
    X = vectorizer.fit_transform(tweets)

    # Crea el clasificador K-NN
    clf = KNeighborsClassifier(n_neighbors=3)

    # Entrena el clasificador
    clf.fit(X, y)
else:
    print('Error al realizar la solicitud')
