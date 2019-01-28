import math as math
import os.path
import re
import sqlite3
import sys
from collections import Counter
from collections import defaultdict

import numpy as np
from nltk import word_tokenize, PorterStemmer
from nltk.stem import WordNetLemmatizer
from scipy.sparse import coo_matrix
from sklearn import metrics, preprocessing
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clustering_metrics import entropy, fmeasure, purity
from kmedoids import KMedoids
import neighbors_metrics
from stopwords import stopwords_long

regex = re.compile("(%s)" % "|".join(
    map(re.escape, ['/', '*', '\'s', '\'ve', '.', ',', ':', ';', '!', '$', '?', '"', '(', ')', '[', ']', '-', '-', '´',
                    '`', '™'])))
algorithms = {
    'aa': neighbors_metrics.adamicadar,
    'j': neighbors_metrics.jaccard,
    'cn': neighbors_metrics.commonneighbours,
    'hpi': neighbors_metrics.hpi,
    'hdi': neighbors_metrics.hdi,
    'salton': neighbors_metrics.salton,
    'pa': neighbors_metrics.pa,
    'lhni': neighbors_metrics.lhni,
    'sorensen': neighbors_metrics.sorensen
}


def clean(string):
    return regex.sub(' ', string)


conn = sqlite3.connect(sys.argv[1])
c = conn.cursor()
corpus = []
labels_true = []
service_number = {}
service_list = []
count = 0
for row in c.execute('SELECT Description,PrimaryCategory,Name,SecondaryCategories FROM Services'):
                     # '  WHERE Name in (SELECT FollowedService FROM ServiceFollowers '
                     # ' GROUP BY FollowedService HAVING COUNT(*)>=2)'):
    label = row[1]
    # if label in ["Other", "Data", "Tools"]:
    #     continue
    corpus.append(clean(row[0] + " " + row[2]))
    # corpus.append(row[0] + " " + row[2])
    # corpus.append(row[0])
    service_list.append(row[2])
    service_number[row[2]] = count
    count += 1

    # secondary = [word for word in row[3].split(",") if word != label and word != ""]
    # if label in ["Other", "Data", "Tools", ""] and len(secondary) != 0:
    #     label = secondary[0]
    labels_true.append(label)

followers = defaultdict(set)
users = defaultdict(set)
for row in c.execute('SELECT Follower,FollowedService FROM ServiceFollowers'):
    svc = row[1]
    if svc not in service_list:
        continue
    followers[service_number.get(row[1])].add(row[0])
    users[row[0]].add(service_number.get(row[1]))

conn.close()


# vectorizer = TfidfVectorizer(sublinear_tf=False,
#                              analyzer='word', lowercase=False,
#                              preprocessor=StringPreprocessorAdapter('english.long'))

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class StemTokenizer(object):
    def __init__(self):
        self.porter = PorterStemmer()

    def __call__(self, doc):
        return [self.porter.stem(t) for t in word_tokenize(doc)]


cosine_file_name = sys.argv[2] + "/cosine_matrix.npy"
if os.path.isfile(cosine_file_name):
    print("Leyendo matriz de coseno")
    coseno = np.load(cosine_file_name)
else:
    vectorizer = TfidfVectorizer(stop_words=stopwords_long,
                                 tokenizer=StemTokenizer())

    # vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5))
    print("Vectorizando corpus")
    tfidf = vectorizer.fit_transform(corpus)
    print("Calculando Coseno")
    coseno = cosine_similarity(tfidf)
    np.save(cosine_file_name, coseno)


def experiments(PORCENTAJE_VECINOS, ALGORITHM, MODELO, normalizar=None):
    vecinos = algorithms[ALGORITHM]

    algoritmos = "coseno"
    if PORCENTAJE_VECINOS in ["boost", "maxsim", "dist"]:
        algoritmos = ALGORITHM + "-" + PORCENTAJE_VECINOS
    elif PORCENTAJE_VECINOS != 0:
        algoritmos = "%s-%.1f" % (ALGORITHM, PORCENTAJE_VECINOS)

    titulo = MODELO + "-" + algoritmos
    if normalizar is not None:
        titulo += "-" + normalizar

    fname = sys.argv[2] + "/" + titulo + ".out"

    if os.path.isfile(fname):
        return

    print(titulo)
    print("-" * 20)

    if PORCENTAJE_VECINOS == 0:
        X = coseno
        if MODELO == "dbscan":
            # Solo sirve para coseno!
            X = 1 - X
    else:
        neighbour_file_name = sys.argv[2] + "/" + ALGORITHM + ".npy"
        if os.path.isfile(neighbour_file_name):
            NEIGHBOURS = np.load(neighbour_file_name)
        else:
            print("Calculando vecinos")
            NEIGHBOURS = np.zeros((len(service_number), len(service_number)))
            for i in range(0, len(service_number)):
                for j in range(i, len(service_number)):
                    NEIGHBOURS[i][j] = vecinos(followers, users, i, j)
                    if i != j:
                        NEIGHBOURS[j][i] = NEIGHBOURS[i][j]
            np.save(neighbour_file_name, NEIGHBOURS)

        if normalizar is not None:
            print("Normalizando Vecinos")
            if normalizar == 'minmax':
                NEIGHBOURS = preprocessing.minmax_scale(NEIGHBOURS)
            elif normalizar == 'scale':
                NEIGHBOURS = preprocessing.scale(NEIGHBOURS)
            elif normalizar == 'robust':
                NEIGHBOURS = preprocessing.robust_scale(NEIGHBOURS)
            elif normalizar == 'softmax':
                NEIGHBOURS = np.exp(NEIGHBOURS) / np.sum(np.exp(NEIGHBOURS), axis=1, keepdims=True)
            elif normalizar == 'matrixminmax':
                NEIGHBOURS = (NEIGHBOURS - np.min(NEIGHBOURS)) / (np.max(NEIGHBOURS) - np.min(NEIGHBOURS))
            elif normalizar == 'matrixmax':
                NEIGHBOURS = NEIGHBOURS / np.max(NEIGHBOURS)
        if MODELO == "dbscan":  # Si es distancia
            if normalizar is not None:
                NEIGHBOURS = 1 - NEIGHBOURS
            else:
                NEIGHBOURS = - NEIGHBOURS
            X = (1 - PORCENTAJE_VECINOS) * (1 - coseno) + PORCENTAJE_VECINOS * NEIGHBOURS
        else:  # Si es afinidad
            if PORCENTAJE_VECINOS == "boost":
                X = np.multiply(coseno, NEIGHBOURS)
            elif PORCENTAJE_VECINOS == "maxsim":
                X = np.maximum(coseno, NEIGHBOURS)
            elif PORCENTAJE_VECINOS == "dist":
                NEIGHBOURS_SORTED = np.argsort(np.argsort(NEIGHBOURS))
                COSINE_SORTED = np.argsort(np.argsort(coseno))
                POS_BOOST = np.log(1 / (1 + np.abs(NEIGHBOURS_SORTED - COSINE_SORTED)))
                X = POS_BOOST
            else:
                X = (1 - PORCENTAJE_VECINOS) * coseno + PORCENTAJE_VECINOS * NEIGHBOURS

    print("Generando Modelo")

    if MODELO == 'kmedoids':
        model = KMedoids(n_clusters=1500).fit(X)
    if MODELO == 'kmedoids470':
        model = KMedoids(n_clusters=470).fit(X)
    elif MODELO == 'ap':
        model = AffinityPropagation(affinity='precomputed').fit(X)
    elif MODELO == 'dbscan':
        model = DBSCAN(metric='precomputed').fit(X)

    labels = model.labels_

    clusters = defaultdict(list)
    for index, classif in enumerate(labels):
        clusters[classif].append(index)

    n_clusters_ = len(clusters)

    info = ""
    info += 'Clusters: %d\n' % n_clusters_
    # info += 'Cohesiveness: %0.3f\n' % cohesiveness(X, labels)
    info += 'Entropy: %0.3f\n' % entropy(labels_true, labels)
    info += "Homogeneity: %0.3f\n" % metrics.homogeneity_score(labels_true, labels)
    info += "Completeness: %0.3f\n" % metrics.completeness_score(labels_true, labels)
    info += "V-measure: %0.3f\n" % metrics.v_measure_score(labels_true, labels)
    info += 'Purity: %0.3f\n' % purity(labels_true, labels)
    info += "F-Measure: %0.3f\n" % fmeasure(labels_true, labels)
    info += "Adjusted Rand Index: %0.3f\n" % metrics.adjusted_rand_score(labels_true, labels)
    info += "Adjusted Mutual Information: %0.3f\n" % metrics.adjusted_mutual_info_score(labels_true, labels)

    clustersize = Counter(labels)

    salida = open(fname, 'w', encoding='UTF-8')

    print(info)

    salida.write(titulo + "\n")
    for cluster, services in clusters.items():
        countcat = Counter([labels_true[svc] for svc in services])
        max_key, num = countcat.most_common(1)[0]
        salida.write("%i (%s - %i/%i): %s \n" % (
            cluster, max_key, num, clustersize[cluster], ",".join([service_list[svc] for svc in services])))
    salida.write("-" * 20 + "\n")
    salida.write(info)
    salida.close()


alg_list = ['cn']
# for model in modelos:
#     experiments(0.0, 'aa', model)  # aa no se usa, es solo un placeholder

modelos = ['ap', 'kmedoids', 'kmedoids470']
porcentajes = [1.0]  # "dist", "boost", "maxsim", 1.0, .2, .4, .6, .8

for model in modelos:
    for alg in alg_list:
        for p in porcentajes:
            experiments(p, alg, model);
            # if p != 1.0:
            # try:
            #     experiments(p, alg, model, 'softmax')
            # except ValueError:
            #     print("Error con softmax")
            #
            # experiments(p, alg, model, 'scale')
            # experiments(p, alg, model, 'robust')
            # experiments(p, alg, model, 'minmax')
            # experiments(p, alg, model, 'matrixminmax')
            # experiments(p, alg, model, 'matrixmax')
