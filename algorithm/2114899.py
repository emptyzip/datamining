import mlxtend
import pandas as pd
from sklearn.datasets import load_iris
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

transactions = []
df = pd.read_csv("groceries - groceries.csv")
df = df.drop(df.columns[0], axis=1)
transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

num=0
while True:
  if num == 5: break
  else:
    print("[ Student ID: 2114899 ]")
    print("[ Name: 전현빈 ]")
    print()
    print("1. Frequent itemsets")
    print("2. Association rules")
    print("3. K-means clustering")
    print("4. Hierarchial clustering")
    print("5. Quit")
    num = int(input('> '))

    if num == 1:
      ms = input("Enter minimum support: ")
      print()

      te = TransactionEncoder()
      te_ary = te.fit(transactions).transform(transactions)
      df_trans = pd.DataFrame(te_ary, columns=te.columns_)

      frequent_itemsets = apriori(df_trans, min_support=float(ms), use_colnames=True)
      print(frequent_itemsets)
      print()

    elif num == 2:
      from mlxtend.frequent_patterns import association_rules
      
      ms = float(input("Enter minimum support: "))
      mc = float(input("Enter minimum confidence: "))
      print()

      te = TransactionEncoder()
      te_ary = te.fit(transactions).transform(transactions)
      df_trans = pd.DataFrame(te_ary, columns=te.columns_)
      
      frequent_itemsets = apriori(df_trans, min_support=ms, use_colnames=True)
      
      rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=mc)
      print(rules[['antecedents', 'consequents', 'support', 'confidence']])
      print()
      
    elif num == 3:
      from sklearn.cluster import KMeans
      import numpy as np

      n = int(input("Enter n_cluster: "))

      iris_data = load_iris().data
    
      result = KMeans(n_clusters=n, random_state=0).fit(iris_data)
      print(result.labels_)
      print()

    elif num == 4:
      from sklearn.cluster import AgglomerativeClustering
      
      n = int(input("Enter n_cluster: "))
      linkage = input("Enter linkage: ")

      iris_data = load_iris().data

      result = AgglomerativeClustering(n_clusters=n, linkage=linkage).fit(iris_data)
      print(result.labels_)
      print()
