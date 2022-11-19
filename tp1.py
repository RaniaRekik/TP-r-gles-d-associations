import itertools

import numpy as numpy
import pandas as pd
from mlxtend.frequent_patterns import association_rules

from mlxtend.frequent_patterns import apriori

""" 1- Créer Un DataFrame en utilisant les données de fichier ‘’market_basket.txt’’
qui contient les identifiants des caddies et les produits associées"""
DataFrame = pd.read_table("market_basket.txt")

# 2-Afficher les 10 premières lignes du DataFrame.
print(DataFrame.head(10))
# 3-Afficher les dimensions du dataframe.
print(DataFrame.shape)

#a = DataFrame.value_counts(["ID", "Product"])
""" 4-Écrire un script python qui permet de Construire un table binaire indiquant la
présence de chaque produit au niveau des caddies (True:1 si le produit est
présent dans le caddie et 0 dans le cas réciproque)."""

rows = DataFrame["ID"].unique()
columns = DataFrame["Product"].sort_values().unique()
matrix = numpy.zeros((len(rows), len(columns)))
cpt = 0

for i in rows:
    data_frame = DataFrame[DataFrame["ID"] == i]
    a = 0
    for j in columns:
        if data_frame["Product"].str.contains(j).any():
            matrix[cpt][a] = 1
        a = a + 1
    cpt = cpt + 1

data_frame = pd.DataFrame(matrix, index=rows, columns=columns)
print("q4", data_frame)

""" 5-Tester la bibliothèque pandas.crosstab pour construire la table binaire et
vérifier que vous avez les mêmes résultats de votre script."""
TableBinaire= pd.crosstab(DataFrame.ID,DataFrame.Product)
print("q5",TableBinaire)

# 6-Afficher les 30 premières transactions et les 3 premiers produits.
q6 = TableBinaire.iloc[:30,:3]
print(q6)

""" 7-Écrire un script python de la fonction a_priori() qui permet l’extraction des
#itemsets les plus fréquents. ( on définit un min_supp=0.025 et un longueur
#maximum de 4 produits) """

ItemsetFrequents = apriori(TableBinaire,min_support=0.025,max_len=4,use_colnames=True)

max = 2
def extraire_itemset(data_frame, n):
    return list(itertools.combinations(data_frame, n))

def support(df, subsets, n):
    tab = []
    k = 0
    for i in subsets:
        subset = df[list(i)]
        w = subset[subset.sum(axis="columns") == n].count()
        tab.append(w[0])
        k = k + 1
    return tab

support_minimal = 0.025 * len(rows)

for i in range(1, max + 1):
    itemset = extraire_itemset(columns, i)
    Support = support(data_frame, itemset, i)
    ff = pd.DataFrame(itemset)
    sup = pd.DataFrame(Support)
    sup.columns = ["Support"]
    frame = [ff, sup]
    F = pd.concat(frame, axis=1)
    C = F[F["Support"] > support_minimal]
    print("C", i)
    print(C)


def regle(item):

    it = item[:-1]
    for i in range(1, max):
        List = extraire_itemset(it, i)
        permis = List[:-i]
        cc = List[i:]
        support_1 = support(data_frame, permis, i)
        supports = support(data_frame, [it], max)
    return (permis, cc, numpy.array(supports) / numpy.array(support_1))


resultats = []
compteur = 0

for i in C.index:
    compteur = compteur + 1
    ligne = list(C.loc[i, :])
    premis, Ccl, Confidence = regle(ligne)
    print("régle :", premis, " : ", Ccl, " confidence:", Confidence)

from mlxtend.frequent_patterns import apriori, association_rules

items_frequents = apriori(data_frame, min_support=0.025, use_colnames=True)

regles = association_rules(items_frequents, metric="lift", min_threshold=1)
regles = regles.sort_values(["confidence", "lift"], ascending=[False, False])
print("q7", regles.head())



""" 9-Ecrire une fonction is_inclus() qui permet de vérifier si un sous-ensemble
items est inclus dans l’ensemble x."""
def is_inclus(x,items):
    return items.issubset(x)

# 10-Afficher les itemsets comprenant le produit ‘Aspirin’
q10 = numpy.where(ItemsetFrequents.itemsets.apply(is_inclus,items={'Aspirin'}))
print(q10)

# 11-Afficher les itemsets contenant Aspirin et Eggs.
print(ItemsetFrequents[ItemsetFrequents['itemsets'].ge({'Aspirin','Eggs'})])

"""12-Nous produisons les règles à partir des itemsets fréquents. Elles peuvent
être très nombreuses,nous en limitons la prolifération en définissant un seuil
minimal (min_threshold = 0.75) sur une mesure d’intérêt, en l’occurrence la
confiance dans notre exemple (metric = ‘’confidence’’).
Utiliser la bibliothèque mlxtend.frequent_patterns pour générer les règles
#d’associations. """
regles = association_rules(ItemsetFrequents,metric="confidence",min_threshold=0.75)

# 13-Afficher les 5 premières règles.
print(regles.iloc[:5,:])

""" 14-Filtrer les règles en affichant celles qui présentent un LIFT supérieur ou
#égal à 7."""
myRegles = regles.loc[:,['antecedents','consequents','lift']]
print(myRegles[myRegles['lift'].ge(7.0)])

# 15-Filtrer les règles en affichant celles menant au conséquent {‘2pct_milk’}.
print(myRegles[myRegles['consequents'].eq({'2pct_Milk'})])