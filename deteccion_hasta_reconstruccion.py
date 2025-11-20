
# %%
experimento="experimento_real"
anomaly_proportion=0.6
objeto="zipper"

# %%
import random
random.seed(42)

# %% [markdown]
# ## Distribución de los datos con centroide

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

class Centroid:
    def __init__(self, k):
        self.k = k
        self.centroid = None
        self.threshold = None

    def fit(self, embeddings):
        # Calcular el centroide
        self.centroid = torch.mean(embeddings, dim=0)

        # Calcular las distancias al centroide
        distances = torch.norm(embeddings - self.centroid, dim=1)

        # Calcular la media y desviación estándar de las distancias
        mean_dist = torch.mean(distances)
        std_dist = torch.std(distances)

        # Definir el umbral
        self.threshold = mean_dist + self.k * std_dist

    def predict(self, embeddings):
        if self.centroid is None or self.threshold is None:
            raise ValueError("Debe llamar al método fit antes de predecir.")

        # Calcular las distancias al centroide
        distances = torch.norm(embeddings - self.centroid, dim=1)

        # Detectar anomalías
        predicted_labels = (distances > self.threshold).int()
        return predicted_labels,distances,self.threshold

# %%
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
import random
import os
names=["tus_embeddings.csv","embeddings.csv","dataset_embeddings_encoder.csv","dataset_embeddings_encoder_resnet.csv"]

for name in names:
    # Cargar el dataset desde el archivo CSV
    # ==========================
    # Cargar los Embeddings
    # ==========================
    df = pd.read_csv(name)

    # Mostrar las primeras filas del dataset
    print(df.head(52))
    random.seed(42)  # Para reproducibilidad

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X = df[df.columns[:-1]]  # Todas las columnas excepto la última
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    k=[0.02,0.05,0.1,0.2,0.5,1,5]

    # Mostrar las primeras filas del conjunto de entrenamiento
    print("Conjunto de entrenamiento:")
    print(X_train.head(10))
    print("Conjunto de prueba:")
    print(X_test.head(10))

    print(f"Total de imagenes en el conjunto de entrenamiento: {len(X_train)}")
    print(f"Total de imagenes en el conjunto de prueba: {len(X_test)}")
    
    

    resultados=[]

    best_k=0
    best_value=0


    print("Anomaly Detector")
    for i in k:

        model=Centroid(i)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        model.fit(X_train_tensor)

        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)


        preds,distances,treshold = model.predict(X_val_tensor)




        acc= accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        auc= roc_auc_score(y_val,distances)
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("Area bajo la curva ROC",auc)
        

        if best_value<f1:
            best_k=i
            best_value=f1


    

    
    model=Centroid(best_k)

    # Concatenar las características (X)
    X_train = np.concatenate([X_train, X_val], axis=0)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    model.fit(X_train_tensor)


    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)


    preds,distances,treshold = model.predict(X_test_tensor)


    acc= accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc= roc_auc_score(y_test,distances)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Area bajo la curva ROC",auc)

    resultados.append({
            'k': best_k,
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': auc
        })
    
    folder_name=f"Test/{experimento}/{objeto}/distance/{name}"

    os.makedirs(folder_name, exist_ok=True)
    
    # Guardar resultados en Excel
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel(os.path.join(folder_name, 'metricas_por_k.xlsx'), index=False)


    

    

# %% [markdown]
# ## Método Antonio

# %%
import torch

class GDAOneClassTorch:
    def fit(self, X, threshold_param=1.0):
        """
        Ajusta el modelo GDA para una sola clase usando PyTorch.
        :param X: Tensor de forma (n_samples, n_features)
        """
        self.mu = X.mean(dim=0)
        self.centered = X - self.mu
        self.sigma = torch.matmul(self.centered.T, self.centered) / X.shape[0]     #calcula la matriz de covarianzas

        # Regularización para evitar matriz singular
        epsilon = 1e-3
        self.sigma += epsilon * torch.eye(self.sigma.shape[0])
        self.inv_sigma = torch.inverse(self.sigma)
        self.det_sigma = torch.det(self.sigma)

        scores= self.score(X)


        mean_score = scores.mean()
        std_score = scores.std()
        self.threshold = mean_score - threshold_param * std_score

    def score(self, X):
        scores=[]
        for i in range(len(X)):
            centered = X[i] - self.mu
            tmp = torch.matmul(centered, self.inv_sigma)
            quad_form = (tmp * centered).sum()
            scores.append(-0.5 * quad_form)
        return torch.tensor(scores)

    def predict(self, X):
        scores = self.score(X)
        preds=[] 
        for i in range(len(scores)):
            pred= 1 if scores[i]< self.threshold else 0
            preds.append((i, pred))

        return preds,self.threshold



    def print_parameters(self):
        """
        Imprime el vector de medias y la matriz de covarianza.
        """
        print("Vector de medias (mu):")
        print(self.mu)
        print("\nMatriz de covarianza (sigma):")
        print(self.sigma)

    def print_score(self, X):
      """
      Imprime y retorna la densidad de cada punto en X.
      :param X: Tensor (n_samples, n_features)
      :return: Diccionario {índice: densidad}
      """
      probs = self.score(X)
      scores = dict()

      for i, p in enumerate(probs):
          valor = p.item()
          print(f"Densidad del punto {i}: {valor:.6f}")
          scores[i] = valor

      return scores

# %%
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random

names=["tus_embeddings.csv","embeddings.csv","dataset_embeddings_encoder.csv","dataset_embeddings_encoder_resnet.csv"]

for name in names:

    df = pd.read_csv(name)

    random.seed(42)  # Para reproducibilidad

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X = df[df.columns[:-1]]  # Todas las columnas excepto la última
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Mostrar las primeras filas del conjunto de entrenamiento
    print("Conjunto de entrenamiento:")
    print(X_train.head(10))
    print("Conjunto de prueba:")
    print(X_test.head(10))

    print(f"Total de imagenes en el conjunto de entrenamiento: {len(X_train)}")
    print(f"Total de imagenes en el conjunto de prueba: {len(X_test)}")


    resultados=[]

    
    k=[0.1,0.5,1,2,5,10,20,30,40,50,60,70,80,100]

    best_k=0
    best_value=0

    for i in k:

        model = GDAOneClassTorch()
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        model.fit(X_train_tensor,i)




        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)


        preds_with_index,treshold= model.predict(X_val_tensor)
        '''
        print(model.threshold)
        preds = torch.tensor(preds)
        print(preds.shape)
        
        # Paso 3: imprimir por pantalla (predicción y etiqueta real)
        for i, (p, real) in enumerate(zip(preds[1], y_test)):
            print(f"Ejemplo {i}: Predicción = {p.item()}, Etiqueta real = {real}")


        '''
        # Extraer predicciones en el mismo orden que y_test
        preds = [pred for x, pred in preds_with_index]
        scores=(model.score(X_val_tensor))**2

        f1 = f1_score(y_val.to_numpy(), preds)
        acc = accuracy_score(y_val.to_numpy(), preds)
        auc= roc_auc_score(y_val,scores)
        print("F1 Score:", f1)
        print("Accuracy:", acc)
        print("Area bajo la curva ROC",auc)

        
        if best_value<f1:
            best_k=i
            best_value=f1


        

    model = GDAOneClassTorch()

    # Concatenar las características (X)
    X_train = np.concatenate([X_train, X_val], axis=0)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    model.fit(X_train_tensor,best_k)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)


    preds_with_index,treshold= model.predict(X_test_tensor)
    '''
    print(model.threshold)
    preds = torch.tensor(preds)
    print(preds.shape)
    
    # Paso 3: imprimir por pantalla (predicción y etiqueta real)
    for i, (p, real) in enumerate(zip(preds[1], y_test)):
        print(f"Ejemplo {i}: Predicción = {p.item()}, Etiqueta real = {real}")


    '''
    # Extraer predicciones en el mismo orden que y_test
    preds = [pred for x, pred in preds_with_index]
    scores=(model.score(X_test_tensor))**2

    f1 = f1_score(y_test.to_numpy(), preds)
    acc = accuracy_score(y_test.to_numpy(), preds)
    auc= roc_auc_score(y_test,scores)
    print("F1 Score:", f1)
    print("Accuracy:", acc)
    print("Area bajo la curva ROC",auc)

    resultados.append({
            'k': best_k,
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': auc
        })

    
    folder_name=f"Test/{experimento}/{objeto}/GDA/{name}"

    os.makedirs(folder_name, exist_ok=True)
    print(f"Creando carpeta: {folder_name}")

    # Guardar resultados en Excel
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel(os.path.join(folder_name, 'metricas_por_k.xlsx'), index=False)



    print(f"Resultados guardados en la carpeta: {folder_name}")

# %% [markdown]
# ## Nuevo Metodo Antonio

# %%
class ZScoreThreshold:
    def fit(self, X):
        """
        Calcula media y desviación estándar por columna.
        X: tensor (n_samples, n_features)
        """
        self.mu = X.mean(dim=0)
        self.std = X.std(dim=0)
        self.std[self.std == 0] = 1e-6  # evita divisiones por cero

    def score(self, X):
        """
        Retorna el número de columnas fuera de lo normal por muestra.
        """
        z_scores = (X - self.mu) / self.std
        return (z_scores.abs() > 1.96).sum(dim=1)  # puedes parametrizar el umbral

    def predict(self, X, percentage=0.1):
        """
        Marca como anomalía si más de un porcentaje de columnas están fuera del estándar.
        
        percentage: porcentaje de columnas que deben estar fuera del estándar para ser considerada anomalía.
        """
        # Calcula el umbral de columnas basado en el porcentaje
        num_columns = int(percentage * X.shape[1])  # Número de columnas según el porcentaje
        
        scores = self.score(X)
        return (scores > num_columns).int(),scores  # Marca como anomalía si supera el umbral de columnas

# %%
from sklearn.metrics import accuracy_score,f1_score

names=["tus_embeddings.csv","embeddings.csv","dataset_embeddings_encoder.csv","dataset_embeddings_encoder_resnet.csv"]

for name in names:

    df = pd.read_csv(name)

    random.seed(42)  # Para reproducibilidad

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    X = df[df.columns[:-1]]  # Todas las columnas excepto la última
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Mostrar las primeras filas del conjunto de entrenamiento
    print("Conjunto de entrenamiento:")
    print(X_train.head(10))
    print("Conjunto de prueba:")
    print(X_test.head(10))

    print(f"Total de imagenes en el conjunto de entrenamiento: {len(X_train)}")
    print(f"Total de imagenes en el conjunto de prueba: {len(X_test)}")

    

    resultados=[]

    k=[0.01,0.05,0.1,0.2,0.25,0.3]

    best_k=0
    best_value=0

    for i in k:

        model=ZScoreThreshold()
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        model.fit(X_train_tensor)

        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        preds,scores = model.predict(X_val_tensor,i)
        print(preds.shape)
        '''
        # Paso 3: imprimir por pantalla (predicción y etiqueta real)
        for i, (p, real) in enumerate(zip(preds, y_test)):
            print(f"Ejemplo {i}: Predicción = {p.item()}, Etiqueta real = {real}")
        '''

        acc= accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        auc= roc_auc_score(y_val,scores)
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("Area bajo la curva ROC",auc)

        if best_value<f1:
            best_k=i
            best_value=f1

    
    model=ZScoreThreshold()

    # Concatenar las características (X)
    X_train = np.concatenate([X_train, X_val], axis=0)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    model.fit(X_train_tensor)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    preds,scores = model.predict(X_test_tensor,best_k)
    
    '''
    # Paso 3: imprimir por pantalla (predicción y etiqueta real)
    for i, (p, real) in enumerate(zip(preds, y_test)):
        print(f"Ejemplo {i}: Predicción = {p.item()}, Etiqueta real = {real}")
    '''

    acc= accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc= roc_auc_score(y_test,scores)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Area bajo la curva ROC",auc)

    resultados.append({
            'k': best_k,
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': auc
        })   
        
    folder_name=f"Test/{experimento}/{objeto}/columnas/{name}"

    os.makedirs(folder_name, exist_ok=True) 
    print(f"Creando carpeta: {folder_name}")   
    # Guardar resultados en Excel
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel(os.path.join(folder_name, 'metricas_por_k.xlsx'), index=False)
