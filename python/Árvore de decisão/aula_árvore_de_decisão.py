# -*- coding: utf-8 -*-
"""Aula_Árvore de decisão.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FcQlg7vGpSOEtB6OE7xy2doGUrg8FfSl
"""

# https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.08-Random-Forests.ipynb#scrollTo=loELzNq5I0Gk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');

# dados gerados e rotulados, montar a árvore
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X, y) # treina o modelo, passando os dados e os rótulos



def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)

visualize_classifier(DecisionTreeClassifier(), X, y)

!pip install graphviz

import graphviz 
from sklearn.tree import export_graphviz

feature_names = tree.columns

dot_data = export_graphviz(tree, out_file=None, 
                         feature_names=feature_names,  
                         class_names=True,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)

graph

"""#O Outro exemplo

"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# !wget https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/diabetes.csv
dataset = pd.read_csv("diabetes.csv")

dataset.head()
# A coluna de resultad( Outcome) indica se a pessoa tem diabetes (1) ou não (0)

dataset.shape

# Dividimos o conjunto de dados em duas variáveis X e y
# X mantém os dados que a árvore de decisão vai usar para aprender e y os rótulos.
features = dataset.drop(["Outcome"], axis=1) #Excluir a coluna de resultado
X = np.array(features)
y = np.array(dataset["Outcome"])

features.columns # verificar que a coluna Outcome foi excluída

"""Conjunto de treinamento e validação
Quando estamos criando um modelo de aprendizado de máquina, é importante ter vários conjuntos de dados:

Conjunto de treinamento
Conjunto de validação
Conjunto de teste
Usamos o conjunto de treinamento para treinar o modelo, o conjunto de validação é usado para verificar como o modelo funciona com diferentes parâmetros e, finalmente, o conjunto de teste é usado para medir o desempenho do modelo, usamos este último conjunto apenas uma vez no final, quando sabemos que o modelo é bom o suficiente. Cada conjunto deve ter diferentes registros para ver como o modelo se comporta com dados que nunca viu.

Às vezes, não temos dados suficientes para dividir o conjunto de dados em três conjuntos diferentes, portanto, usamos o conjunto de treinamento para treinar o modelo e o conjunto de validação para testar diferentes parâmetros e medir o desempenho do modelo.
"""

# Podemos dividir o conjunto de dados com a seguinte função scikit-learn
# Usamos 20% do conjunto de dados para construir o conjunto de validação.
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=0.20)

# Podemos criar um modelo de árvore de decisão com o seguinte código:
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train) # método fit faz o treinamento do modelo

# usou o padrão algoritmo GINI =  p^2 + q^2 - p e q são os mesmos probabilidades do que na fórmula entropia

# ver a profundidade da árvore: 16 -  temos uma alta probabilidade de sobreajuste. Maior q 5 ocorre o sobreajuste
tree.tree_.max_depth

# Usaremos o conjunto de validação para medir o desempenho do modelo
validation_prediction = tree.predict(X_val)
training_prediction = tree.predict(X_train)

print('Accuracy training set: ', accuracy_score(y_true=y_train, y_pred=training_prediction))
print('Accuracy validation set: ', accuracy_score(y_true=y_val, y_pred=validation_prediction))

"""O conjunto de treinamento tem uma precisão de 100% e o conjunto de validação tem uma precisão de 79%, isso significa que temos um problema de overfitting , o modelo é muito bom em prever os registros do conjunto de treinamento, mas não é tão bom com o conjunto de validação."""

!pip install graphviz

import graphviz 
from sklearn.tree import export_graphviz

feature_names = features.columns

dot_data = export_graphviz(tree, out_file=None, 
                         feature_names=feature_names,  
                         class_names=True,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)

graph

"""**Criando outro exemplo**"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

!wget https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/diabetes.csv

dataset = pd.read_csv("diabetes.csv")

# Dividimos o conjunto de dados em duas variáveis X e y
# X mantém os dados que a árvore de decisão vai usar para aprender e y os rótulos.
features = dataset.drop(["Outcome"], axis=1) #Excluir a coluna de resultado
X = np.array(features)
y = np.array(dataset["Outcome"])

# Podemos dividir o conjunto de dados com a seguinte função scikit-learn
# Usamos 20% do conjunto de dados para construir o conjunto de validação.
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, test_size=0.20)

#tree = DecisionTreeClassifier(min_samples_leaf=10, max_depth=8, min_samples_split=50)
tree = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4, min_samples_split=20)

tree.fit(X_train, y_train)

validation_prediction = tree.predict(X_val)
training_prediction = tree.predict(X_train)

print('Exactitud training data: ', accuracy_score(y_true=y_train, y_pred=training_prediction))
print('Exactitud validation data: ', accuracy_score(y_true=y_val, y_pred=validation_prediction))

!pip install graphviz

import graphviz 
from sklearn.tree import export_graphviz

feature_names = features.columns
dot_data = export_graphviz(tree, out_file=None, 
                         feature_names=feature_names,  
                         class_names=True,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)

graph