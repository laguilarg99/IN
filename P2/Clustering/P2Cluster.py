# -*- coding: utf-8 -*-
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from math import floor
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter
from unicodedata import normalize
import pandas as pd
import seaborn
import numpy as np
import calendar
import matplotlib.pyplot as plt
import warnings


def to_matrix(df, columns=[]):
    """Devuelve los atributos seleccionados como valores"""
    return df[columns].dropna().values

def norm(data):
    """Normaliza una serie de datos"""
    return (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

def measures_silhoutte_calinski(data, labels):
    """
    Devuelve el resultado de evaluar los clusters de data asociados con labels.
    
    Parámetros:
    
    - data vector de datos ya normalizados.
    - labels: etiquetas.
    """
    # Hacemos una muestra de sólo el 20% porque son muchos elementos
    muestra_silhoutte = 0.2 if (len(data) > 10000) else 1.0
    silhouette = silhouette_score(data, labels, metric='euclidean', sample_size=int(floor(data.shape[0]*muestra_silhoutte)))
    calinski = calinski_harabasz_score(data, labels)
    return silhouette, calinski

def print_measure(measure, value):
    """
    Muestra el valor con un número fijo de decimales
    """
    print("{}: {:.3f}".format(measure, value))


def pairplot(df, columns, labels):
    """
    Devuelve una imagen pairplot.

    Parámetros:

    - df: dataframe
    - columns: atributos a considerar
    - labels: etiquetas
    """
    df_plot = df.loc[:,columns].dropna()
    df_plot['classif'] = labels
    seaborn.pairplot(df_plot, hue='classif', palette='Paired')


def denorm(data, df):
    """
    Permite desnormalizar
    """
    return data*(df.max(axis=0)-df.min(axis=0))+df.min(axis=0)


def visualize_centroids(centers, data, columns):
    """
    Visualiza los centroides.

    Parametros:

    - centers: centroides.
    - data: listado de atributos.
    - columns: nombres de los atributos.
    """
    df_centers = pd.DataFrame(centers,columns=columns)
    centers_desnormal=denorm(centers, data)
    hm = seaborn.heatmap(df_centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize = 8)
    # estas tres lineas las he añadido para evitar que se corten la linea superior e inferior del heatmap
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    hm.figure.tight_layout()
    return hm



fichero = "data/accidentes_2013.csv"
data = pd.read_csv(fichero,na_values=['?']) 
data = data.dropna()

#ACCIDENTES POR MES
accidentes_mes = data.groupby(data['MES']).count()

accidentes_mes.index = [calendar.month_name[x] for x in range (1,13)]

print(accidentes_mes['HORA'])

accidentes_mes.plot(kind='bar', figsize=(12,7), color='blue', alpha=0.5)
plt.title('Accidentes en 2013', fontsize=20)
plt.xlabel('Meses',fontsize=16)
plt.ylabel('Numero de accidentes', fontsize=16)
plt.legend().remove()
plt.savefig('AccidentesMes.png')
plt.show()

#ACCIDENTES POR DIA
accidentes_dia = data.groupby(data['DIASEMANA']).count()

accidentes_dia.index = [calendar.day_name[x] for x in range (0,7)]

print(accidentes_dia['HORA'])

accidentes_dia.plot(kind='bar', figsize=(12,7), color='blue', alpha=0.5)
plt.title('Accidentes en 2013', fontsize=20)
plt.xlabel('Dias de la semana',fontsize=16)
plt.ylabel('Numero de accidentes', fontsize=16)
plt.legend().remove()
plt.savefig('AccidentesDia.png')
plt.show()


# Caso de estudio 1
# Colisiones de vehiculos
# Buen Tiempo / Viento Fuerte

#Choque de vehiculos
vehiculos = data[data['TOT_VEHICULOS_IMPLICADOS'] > 1]

#Dias con BuenTiempo
data_BuenTiempo = vehiculos[vehiculos['FACTORES_ATMOSFERICOS']=='BUEN TIEMPO']

#Son los únicos atributos numéricos que aportan un valor real a la segmentación
atributos = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

#1.1 Buen Tiempo KMeans
data_u = to_matrix(data_BuenTiempo,atributos)

#Normalizamos
data_norm_BuenTiempo = norm(data_u)

#KMeans

modelKmeans = KMeans()
visualizer = KElbowVisualizer(modelKmeans, k=(2,30), timings = True)
visualizer.fit(data_norm_BuenTiempo)
visualizer.show()


results = KMeans(n_clusters=visualizer.elbow_value_, random_state = 0).fit(data_norm_BuenTiempo)
labels_kmeans = results.labels_
centroids = results.cluster_centers_

#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_BuenTiempo, labels_kmeans)

print("Buen Tiempo Kmeans silhouette: {:3f}".format(silhouette))
print("Buen Tiempo Kmeans calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_BuenTiempo, atributos)
plt.savefig("centroidesBuenTiempokmean.png")
plt.show()
pairplot(data_BuenTiempo, atributos, labels_kmeans)
plt.savefig("PairplotBuenTiempokmean.png")
plt.show()

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#1.2 Buen Tiempo Ward
modelWard = AgglomerativeClustering()
visualizer = KElbowVisualizer(modelWard, k=(2,30), timings = True)
visualizer.fit(data_norm_BuenTiempo)
visualizer.show()

#Algoritmo Ward
results = AgglomerativeClustering(n_clusters = visualizer.elbow_value_, affinity = 'euclidean', linkage = 'ward').fit(data_norm_BuenTiempo)
labels_ward = results.labels_

#Calcular cluster a mano
data_centro = pd.DataFrame(data_norm_BuenTiempo)
data_centro['cluster'] = labels_ward
data_centroides = data_centro.groupby('cluster').mean()
centroids = data_centroides.values


#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_BuenTiempo, labels_ward)

print("Buen Tiempo Ward silhouette: {:3f}".format(silhouette))
print("Buen TIempo Ward calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_BuenTiempo, atributos)
plt.savefig("centroidesBuenTiempoWard.png")
plt.show()
pairplot(data_BuenTiempo, atributos, labels_ward)
plt.savefig("PairplotBuenTiempoWard.png")
plt.show()


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

#Choque de vehiculos
vehiculos = data[data['TOT_VEHICULOS_IMPLICADOS'] > 1]

#Dias con LLuvia Fuerte
data_VientoFuerte = vehiculos[vehiculos['FACTORES_ATMOSFERICOS']=='VIENTO FUERTE']

#Son los únicos atributos numéricos que aportan un valor real a la segmentación
atributos = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

#2.1 Viento Fuerte KMeans
data_u = to_matrix(data_VientoFuerte,atributos)

#Normalizamos
data_norm_VientoFuerte = norm(data_u)

#KMeans
Sum_of_squared = []

i_range = range(2,20)
for i in i_range:
    km = KMeans(n_clusters= i, random_state = 0).fit(data_norm_VientoFuerte)
    Sum_of_squared.append(km.inertia_)

plt.plot(i_range, Sum_of_squared, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


results = KMeans(n_clusters= 8, random_state = 0).fit(data_norm_VientoFuerte)
labels_kmeans = results.labels_
centroids = results.cluster_centers_

#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_VientoFuerte, labels_kmeans)

print("Viento fuerte Kmeans silhouette: {:3f}".format(silhouette))
print("Viento fuerte Kmeans calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_VientoFuerte, atributos)
plt.savefig("centroidesVientoFuerteKmeans.png")
plt.show()
pairplot(data_VientoFuerte, atributos, labels_kmeans)
plt.savefig("pairplotVientoFuerteKmeans.png")
plt.show()


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#2.2 Viento Fuerte Ward
modelWard = AgglomerativeClustering()
visualizer = KElbowVisualizer(modelWard, k=(2,30), timings = True)
visualizer.fit(data_norm_VientoFuerte)
visualizer.show()

#Algoritmo Ward
results = AgglomerativeClustering(n_clusters = visualizer.elbow_value_, affinity = 'euclidean', linkage = 'ward').fit(data_norm_VientoFuerte)
labels_ward = results.labels_

#Calcular cluster a mano
data_centro = pd.DataFrame(data_norm_VientoFuerte)
data_centro['cluster'] = labels_ward
data_centroides = data_centro.groupby('cluster').mean()
centroids = data_centroides.values


#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_VientoFuerte, labels_ward)

print("Viento Fuerte Ward silhouette: {:3f}".format(silhouette))
print("Viento Fuerte Ward calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_VientoFuerte, atributos)
plt.savefig("centroidesVientoFuerteWard.png")
plt.show()
pairplot(data_VientoFuerte, atributos, labels_ward)
plt.savefig("pairplotVientoFuerteWard.png")
plt.show()

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

# Caso de estudio 2
# Andalucia / Madrid 

#Accidentes en andalucia
data_Andalucia = data[data['COMUNIDAD_AUTONOMA']== 'Andalucía']

#Son los únicos atributos numéricos que aportan un valor real a la segmentación
atributos = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

#3.1 Andalucía Kmeans
data_u = to_matrix(data_Andalucia,atributos)

#Normalizamos
data_norm_Andalucia = norm(data_u)

#KMeans

modelKmeans = KMeans()
visualizer = KElbowVisualizer(modelKmeans, k=(2,30), timings = True)
visualizer.fit(data_norm_Andalucia)
visualizer.show()


results = KMeans(n_clusters=visualizer.elbow_value_, random_state = 0).fit(data_norm_Andalucia)
labels_kmeans = results.labels_
centroids = results.cluster_centers_

#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_Andalucia, labels_kmeans)

print("Andalucia Kmeans silhouette: {:3f}".format(silhouette))
print("Andalucia Kmeans calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_Andalucia, atributos)
plt.savefig("centroidesAndaluciaKmean.png")
plt.show()
pairplot(data_Andalucia, atributos, labels_kmeans)
plt.savefig("centroidesAndaluciaKmean.png")
plt.show()

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#3.2 Andalucia Ward
modelWard = AgglomerativeClustering()
visualizer = KElbowVisualizer(modelWard, k=(2,30), timings = True)
visualizer.fit(data_norm_Andalucia)
visualizer.show()

#Algoritmo Ward
results = AgglomerativeClustering(n_clusters = visualizer.elbow_value_, affinity = 'euclidean', linkage = 'ward').fit(data_norm_Andalucia)
labels_ward = results.labels_

#Calcular cluster a mano
data_centro = pd.DataFrame(data_norm_Andalucia)
data_centro['cluster'] = labels_ward
data_centroides = data_centro.groupby('cluster').mean()
centroids = data_centroides.values


#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_Andalucia, labels_ward)

print("Andalucia Ward silhouette: {:3f}".format(silhouette))
print("Andalucia Ward calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_Andalucia, atributos)
plt.savefig("centroidesAndaluciaWard.png")
plt.show()
pairplot(data_Andalucia, atributos, labels_ward)
plt.savefig("pairplotAndaluciaWard.png")
plt.show()


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#2.2 Madrid

#Accidentes en Madrid
data_madrid = data[data['PROVINCIA'] == 'Madrid']

#Son los únicos atributos numéricos que aportan un valor real a la segmentación
atributos = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']

#4.1 Madrid Kmeans
data_u = to_matrix(data_madrid,atributos)

#Normalizamos
data_norm_madrid = norm(data_u)

#KMeans

modelKmeans = KMeans()
visualizer = KElbowVisualizer(modelKmeans, k=(2,30), timings = True)
visualizer.fit(data_norm_madrid)
visualizer.show()


results = KMeans(n_clusters=visualizer.elbow_value_, random_state = 0).fit(data_norm_madrid)
labels_kmeans = results.labels_
centroids = results.cluster_centers_

#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_madrid, labels_kmeans)

print("Madrid Kmeans silhouette: {:3f}".format(silhouette))
print("Madrid Kmeans calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_madrid, atributos)
plt.savefig("centroidesMadridKmean.png")
plt.show()
pairplot(data_madrid, atributos, labels_kmeans)
plt.savefig("centroidesMadridKmean.png")
plt.show()


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#4.2 Madrid Ward
modelWard = AgglomerativeClustering()
visualizer = KElbowVisualizer(modelWard, k=(2,30), timings = True)
visualizer.fit(data_norm_madrid)
visualizer.show()

#Algoritmo Ward
results = AgglomerativeClustering(n_clusters = visualizer.elbow_value_, affinity = 'euclidean', linkage = 'ward').fit(data_norm_madrid)
labels_ward = results.labels_

#Calcular cluster a mano
data_centro = pd.DataFrame(data_norm_madrid)
data_centro['cluster'] = labels_ward
data_centroides = data_centro.groupby('cluster').mean()
centroids = data_centroides.values


#Obtener Medidas
silhouette, calinski = measures_silhoutte_calinski(data_norm_madrid, labels_ward)

print("Madrid Ward silhouette: {:3f}".format(silhouette))
print("Madrid Ward calinsky: {:3f}".format(calinski))

#Visualizacion
visualize_centroids(centroids, data_norm_madrid, atributos)
plt.savefig("centroidesMadridWard.png")
plt.show()
pairplot(data_madrid, atributos, labels_ward)
plt.savefig("centroidesMadridWard.png")
plt.show()

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
