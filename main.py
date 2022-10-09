import pandas as pd
import numpy as np
import re
import razdel
import pymorphy2
from sklearn.decomposition import PCA
import torch
import nltk
from navec import Navec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestCentroid
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
import warnings
import pickle

warnings.filterwarnings("ignore")

nltk.download('stopwords')

stop_eng = set(stopwords.words('english') + ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
                                            'h', 'i', 'j', 'k', 'l', 'm', 'n', 
                                            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
stop_rus = set(stopwords.words('russian') + ['а', 'о', 'у', 'ы', 'э', 'я', 'е', 
                                            'ё', 'ю', 'и', 'б', 'в', 'г', 'д', 
                                            'й', 'ж', 'з', 'к', 'л', 'м', 'н', 
                                            'п', 'р', 'с', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ']) 
stop_words = stop_eng.union(stop_rus)

morph = pymorphy2.MorphAnalyzer()

def parse_tokens(text: str) -> list:
    words = re.split(r"\[|, '|'|\]", text)
    return " ".join(words).strip().split("  ")

def get_emb(row, tfidf):
    embds = list()
    if row is not None:
        for i, word in enumerate(row.split()):
            word = word.replace('ё', 'е')
            if word in navec:
                embds.append(navec[word] * tfidf[i])
    
    embds = np.array(embds)

    if len(embds):
        return embds.mean(axis=0)
    else:
        return np.zeros((300))

def tokenize_with_razdel(text: str) -> list:
    """
    Токенизация с помощью модуля razdel
    """
    return [token.text for token in razdel.tokenize(text)]


def lemmatize(tokens: list) -> str:
    words = [morph.parse(token)[0].normal_form for token in tokens]
    return " ".join([word for word in words if word not in stop_words and len(word) > 2])

def str_list(text):
    return list(map(float, text[1:-1].split(', ')))
    
def preprocessing_name(text: str) -> str:
    """
    Токенизация и лемматизация текста
    """
    if isinstance(text, str):
        return lemmatize(tokenize_with_razdel(clean_text(text)))
    return ""

def clean_text(text):
    text = re.sub('[^а-яa-zёА-ЯA-ZЁ-]+', ' ', text.strip())
    text = re.sub(' - |- | -', ' ', text.lower())
    return re.sub('\s+', ' ', text)

data = pd.read_csv('preprocessing/data.csv')
data['tfidf_vectors'] = data['tfidf_vectors'].apply(str_list)
path = 'preprocessing/navec_news_v1_1B_250K_300d_100q.tar'
navec = Navec.load(path)
data['text_vectors'] = data.apply(lambda x: get_emb(x['text_tokens'], x['tfidf_vectors']), axis=1) 
print("vectors")
pca = PCA(n_components=5)
pca.fit(torch.tensor(data['text_vectors']))
reduced_vectors = pca.transform(torch.tensor(data['text_vectors']))
with open("reduced_vect", "wb") as f:
    pickle.dump(reduced_vectors, f)
with open("pca_reduce", "wb") as f:
    pickle.dump(pca, f)
print(f"pca")
cluster = KMeans(n_clusters=25)
print("agg")
clusters_aggl_pca = cluster.fit_predict(torch.tensor(reduced_vectors))
print(clusters_aggl_pca)
data['clusters'] = clusters_aggl_pca
data.to_csv("preprocessing/data.csv", index=False)