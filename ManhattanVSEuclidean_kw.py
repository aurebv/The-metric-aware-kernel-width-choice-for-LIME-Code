import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn import ensemble
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#import lime.lime_tabular
from lime_stability.stability import LimeTabularExplainerOvr
from auxiliary import *


seed = 4
random.seed(seed)


# Experiment: Using default kernel width. 
min_dimension=2
max_dimension=20
step=2
Eu_data = []
Man_data = []

for d in range(min_dimension,max_dimension,step): #Cuantos atributos va a haber en el dataset
  X, y = make_classification(n_samples=500, n_features=d, n_informative = d, n_redundant = 0, n_classes=2, n_clusters_per_class=1, random_state=seed)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=seed) #En cada iteracion redefinimos el data set que  usa la funcion Explication study
  n=X_test.shape[0]
  classifier = ensemble.RandomForestClassifier()
  classifier.fit(X_train, y_train);
  explainer = LimeTabularExplainerOvr(X_train, verbose = False, discretize_continuous=True, random_state = seed) #kernel width por defecto 
  #Euclidean calculations
  R_array = np.zeros(n) 
  CSI_array = np.zeros(n)
  VSI_array = np.zeros(n)
  for i in range(n): #Recorre las instancias del test set
    r,csi,vsi = Explication_study(i,d,4000,"euclidean",explainer,classifier, X_test)
    R_array[i]=r #guardamos los datos en los array
    CSI_array[i]=csi
    VSI_array[i]=vsi
  Eu_data.append((np.mean(R_array),np.mean(CSI_array),np.mean(VSI_array))) #Metemos las medias de los array en los datos que pasamos a la funcion para graficar
  #Manhattan calculations
  R_array = np.zeros(n) 
  CSI_array = np.zeros(n)
  VSI_array = np.zeros(n)
  for i in range(n): #Repetimos el proceso con manhattan
    r,csi,vsi = Explication_study(explainer,i,d,4000,"manhattan",explainer,classifier, X_test)
    R_array[i]=r
    CSI_array[i]=csi
    VSI_array[i]=vsi
  Man_data.append((np.mean(R_array),np.mean(CSI_array),np.mean(VSI_array)))
  print(d)

Plot_data(Eu_data,Man_data,min_dimension,max_dimension,step,Eu_data,Man_data) #graficamos 



### Experiment: Custom kernel width

min_dimension=2
max_dimension=20
step=2
Eu_data = []
Man_data = []

for d in range(min_dimension,max_dimension,step): #Cuantos atributos va a haber en el dataset
  X, y = make_classification(n_samples=500, n_features=d, n_informative = d, n_redundant = 0, n_classes=2, n_clusters_per_class=1, random_state=seed)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=seed)
  n=X_test.shape[0]
  classifier = ensemble.RandomForestClassifier()
  classifier.fit(X_train, y_train);
  #Instanciamos los dos explainers
  explainer_eu = LimeTabularExplainerOvr(X_train, verbose = False, discretize_continuous=True, random_state = seed) #Default kernel 
  explainer_man = LimeTabularExplainerOvr(X_train, kernel_width= d*0.75, verbose = False, discretize_continuous=True, random_state = seed) #Custom kernel
  #Euclidean calculations
  R_array = np.zeros(n) 
  CSI_array = np.zeros(n)
  VSI_array = np.zeros(n)
  for i in range(n):
    r,csi,vsi =Explication_study(i,d,4000,"euclidean",explainer_eu,classifier, X_test)
    R_array[i]=r
    CSI_array[i]=csi
    VSI_array[i]=vsi
  Eu_data.append((np.mean(R_array),np.mean(CSI_array),np.mean(VSI_array)))
  #Manhattan calculations
  R_array = np.zeros(n) 
  CSI_array = np.zeros(n)
  VSI_array = np.zeros(n)
  for i in range(n):
    r,csi,vsi = Explication_study(i,d,4000,"manhattan",explainer_man,classifier, X_test)
    R_array[i]=r
    CSI_array[i]=csi
    VSI_array[i]=vsi
  Man_data.append((np.mean(R_array),np.mean(CSI_array),np.mean(VSI_array)))
  print(d)

Plot_data(Eu_data,Man_data,min_dimension,max_dimension,step,Eu_data,Man_data)