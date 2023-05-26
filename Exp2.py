import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets,ensemble
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#import lime.lime_tabular
from lime_stability.stability import LimeTabularExplainerOvr
from auxiliary import *
from tabulate import tabulate

seed = 4
random.seed(seed)

data = datasets.load_breast_cancer() #Dimensionality:30; Samples:569
X = data['data']
y = data['target']
features = data['feature_names']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=50)
classifier = ensemble.RandomForestClassifier()
classifier.fit(X_train, y_train);
n=X_test.shape[0]

#Experiment-----------

explainer_default = LimeTabularExplainerOvr(X_train, verbose = False, discretize_continuous=True, random_state = seed) #kernel width por defecto 
explainer_custom = LimeTabularExplainerOvr(X_train, kernel_width= 30*0.75, verbose = False, discretize_continuous=True, random_state = seed) #Custom kernel

Eu_attributes=[]
Man_attributes_default=[]
Man_attributes_custom=[]

#Euclidean calculations
R_array = np.zeros(n) 
CSI_array = np.zeros(n)
for i in range(n): #test set instances
  r,csi,vsi,used = Explication_study(explainer_default,i,30,4000,"euclidean",classifier, X_test)
  R_array[i]=r 
  CSI_array[i]=csi
  Eu_attributes.append(used)
Eu_r_mean = np.mean(R_array)
Eu_r_var = np.var(R_array)
Eu_CSI_mean = np.mean(CSI_array)
Eu_CSI_var = np.var(CSI_array)
print("Euclidean")

#Manhattan with default kernel width calculations
R_array = np.zeros(n) 
CSI_array = np.zeros(n)
for i in range(n): 
  r,csi,vsi,used = Explication_study(explainer_default,i,30,4000,"manhattan",classifier, X_test)
  R_array[i]=r 
  CSI_array[i]=csi
  Man_attributes_default.append(used)
Man_r_mean = np.mean(R_array)
Man_r_var = np.var(R_array)
Man_CSI_mean = np.mean(CSI_array)
Man_CSI_var = np.var(CSI_array)
print("Manhattan_default")

#Manhattan with custom kernel width calculations
R_array = np.zeros(n) 
CSI_array = np.zeros(n)
for i in range(n): 
  r,csi,vsi,used = Explication_study(explainer_custom,i,30,4000,"manhattan",classifier, X_test)
  R_array[i]=r 
  CSI_array[i]=csi
  Man_attributes_custom.append(used)
Man_r_mean_custom = np.mean(R_array)
Man_r_var_custom = np.var(R_array)
Man_CSI_mean_custom = np.mean(CSI_array)
Man_CSI_var_custom = np.var(CSI_array)
print("Manhattan-custom")


np.savetxt('Tabla_previa.txt',[ ["Euclidean_r",Eu_r_mean,Eu_r_var],
                                ["Manhattan_r_default",Man_r_mean,Man_r_var],
                                ["Manhattan_r_custom",Man_r_mean_custom,Man_r_var_custom]
                                ["Euclidean_csi",Eu_CSI_mean,Eu_CSI_var],
                                ["Manhattan_csi_default",Man_CSI_mean,Man_CSI_var],
                                ["Manhattan_csi_custom",Man_CSI_mean_custom,Man_CSI_var_custom]])
np.savetxt('Euclidean_attributes.txt',Eu_attributes)
np.savetxt('Manhattan_attributes_default.txt',Man_attributes_default)
np.savetxt('Manhattan_attributes_custom.txt',Man_attributes_custom)

#Definning the distance to study explanation similarity (Eduardo Paluzo)
def distance_by_position(list1,list2):

  ds = list()

  l = len(list1)

  for i in range(l):

    ds.append(np.abs(i-np.argwhere([j==list1[i] for j in list2])[0]))

  return np.concatenate(ds)

Distances_Eu_vs_Man_default=[]
Distances_Eu_vs_Man_custom=[]

for i in range(n):
  Distances_Eu_vs_Man_default.append(distance_by_position(Eu_attributes[i],Man_attributes_default[i]))
  Distances_Eu_vs_Man_custom.append(distance_by_position(Eu_attributes[i],Man_attributes_custom[i]))

np.savetxt('Distances_default.txt',Distances_Eu_vs_Man_default)
np.savetxt('Distances_custom.txt',Distances_Eu_vs_Man_custom)


#Similarity study 
Default_means_array=np.zeros(n)
Custom_means_array=np.zeros(n)
for i in range(n):
  Default_means_array[i] = np.mean(Distances_Eu_vs_Man_default[i])
  Custom_means_array[i] = np.mean(Distances_Eu_vs_Man_custom[i])
print("Default: ")
print(np.mean(Default_means_array))
print("Custom: ")
print(np.mean(Custom_means_array))