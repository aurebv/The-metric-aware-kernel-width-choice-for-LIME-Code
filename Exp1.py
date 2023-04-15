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


#In experiment 1 we study R^2 and CSI values over dimensionality using manhattan or euclidean distance. In each dimension we calculate the
# mean of these values over the test_set, we repeate this proces k times and use this data to graphic the resoults.

k=10
min_dimension=5
max_dimension=40
step=5
Eu_data = [] #Listas con la informacion global
Man_data = []
Man_data_custom = []
#En Eu_data[0] se guarda la info del R^2 y en Eu_data[1] se guarda la info del CSI 

Eu_data_r = [] 
Eu_data_csi = []
Man_data_r = []
Man_data_csi =[]
Man_data_r_custom = []
Man_data_csi_custom = []
#En Eu_data_r[i] se almacena un array con las k medias del R^2 sobre el test_set de k data sets diferentes con dimension min_dim+i*step

Eu_data_r_var=[] 
Eu_data_csi_var=[]
Man_data_r_var=[]
Man_data_csi_var=[]
Man_data_r_var_custom=[]
Man_data_csi_var_custom=[]
#En Eu_data_r_var[i] se guarda un array con las k varianzas del r^2 sobre el test_set 

for d in range(min_dimension,max_dimension,step): #Cuantos atributos va a haber en el dataset
  Eu_array_r_means = np.zeros(k) #Son los arrays que despues guardamos en las listas
  Eu_array_csi_means = np.zeros(k)
  Man_array_r_means = np.zeros(k)
  Man_array_csi_means = np.zeros(k)
  Man_array_r_means_custom = np.zeros(k)
  Man_array_csi_means_custom = np.zeros(k)

  Eu_array_r_vars = np.zeros(k) #Son los arrays que despues guardamos en las listas
  Eu_array_csi_vars = np.zeros(k)
  Man_array_r_vars = np.zeros(k)
  Man_array_csi_vars = np.zeros(k)
  Man_array_r_vars_custom = np.zeros(k)
  Man_array_csi_vars_custom = np.zeros(k)
  for seed in range(k):
    random.seed(seed)
    X, y = make_classification(n_samples=500, n_features=d, n_informative = d, n_redundant = 0, n_classes=2, n_clusters_per_class=1, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=seed) #En cada iteracion redefinimos el data set que  usa la funcion Explication study
    n=X_test.shape[0]
    classifier = ensemble.RandomForestClassifier()
    classifier.fit(X_train, y_train);
    explainer_default = LimeTabularExplainerOvr(X_train, verbose = False, discretize_continuous=True, random_state = seed) #kernel width por defecto 
    explainer_custom = LimeTabularExplainerOvr(X_train, kernel_width= d*0.75, verbose = False, discretize_continuous=True, random_state = seed) #Custom kernel
    #Euclidean calculations with default kw
    R_array = np.zeros(n) 
    CSI_array = np.zeros(n)
    for i in range(n): #Recorre las instancias del test set
        r,csi,vsi,used = Explication_study(explainer_default,i,d,4000,"euclidean",classifier, X_test)
        R_array[i]=r #guardamos los datos en los array
        CSI_array[i]=csi
    Eu_array_r_means[seed] = np.mean(R_array)
    Eu_array_csi_means[seed] = np.mean(CSI_array)
    Eu_array_r_vars[seed] = np.var(R_array)
    Eu_array_csi_vars[seed] = np.var(CSI_array)
    #Manhattan calculations with default kw
    R_array = np.zeros(n) 
    CSI_array = np.zeros(n)
    for i in range(n): #Repetimos el proceso con manhattan
        r,csi,vsi,used = Explication_study(explainer_default,i,d,4000,"manhattan",classifier, X_test)
        R_array[i]=r
        CSI_array[i]=csi
    Man_array_r_means[seed]=np.mean(R_array)
    Man_array_csi_means[seed] = np.mean(CSI_array)
    Man_array_r_vars[seed]=np.var(R_array)
    Man_array_csi_vars[seed] = np.var(CSI_array)
    #Manhattan calculations with custom kw
    R_array = np.zeros(n) 
    CSI_array = np.zeros(n)
    for i in range(n): #Repetimos el proceso
        r,csi,vsi,used = Explication_study(explainer_custom,i,d,4000,"manhattan",classifier, X_test) #custom explainer
        R_array[i]=r
        CSI_array[i]=csi
    Man_array_r_means_custom[seed]=np.mean(R_array)
    Man_array_csi_means_custom[seed] = np.mean(CSI_array)
    Man_array_r_vars_custom[seed]=np.var(R_array)
    Man_array_csi_vars_custom[seed] = np.var(CSI_array)
    
  Eu_data_r.append(Eu_array_r_means) #Añadimos a la lista el array con las medias de los 10 experimentos
  Eu_data_csi.append(Eu_array_csi_means)
  Man_data_r.append(Man_array_r_means)
  Man_data_csi.append(Man_array_csi_means)
  Man_data_r_custom.append(Man_array_r_means_custom)
  Man_data_csi_custom.append(Man_array_csi_means_custom)

  Eu_data_r_var.append(Eu_array_r_vars) #Añadimos a la lista el array con las medias de los 10 experimentos
  Eu_data_csi_var.append(Eu_array_csi_vars)
  Man_data_r_var.append(Man_array_r_vars)
  Man_data_csi_var.append(Man_array_csi_vars)
  Man_data_r_var_custom.append(Man_array_r_vars_custom)
  Man_data_csi_var_custom.append(Man_array_csi_vars_custom)
  print(d)
Eu_data.append(Eu_data_r) #estas asignaciones son para pasar menos argumentos a plot
Eu_data.append(Eu_data_csi)
Man_data.append(Man_data_r)
Man_data.append(Man_data_csi)
Man_data_custom.append(Man_data_r_custom)
Man_data_custom.append(Man_data_csi_custom)

#guardamos los datos
np.savetxt('Euclidean_data_r.txt',Eu_data_r)
np.savetxt('Euclidean_data_csi.txt',Eu_data_csi)
np.savetxt('Manhattan_data_r.txt',Man_data_r)
np.savetxt('Manhattan_data_csi.txt',Man_data_csi)
np.savetxt('Manhattan_data_r_custom.txt',Man_data_r_custom)
np.savetxt('Manhattan_data_csi_custom.txt',Man_data_csi_custom)

np.savetxt('Euclidean_data_r_var.txt',Eu_data_r_var)
np.savetxt('Euclidean_data_csi_var.txt',Eu_data_csi_var)
np.savetxt('Manhattan_data_r_var.txt',Man_data_r_var)
np.savetxt('Manhattan_data_csi_var.txt',Man_data_csi_var)
np.savetxt('Manhattan_data_r_custom_var.txt',Man_data_r_var_custom)
np.savetxt('Manhattan_data_csi_custom_var.txt',Man_data_csi_var_custom)


Plot_data(Eu_data,Man_data,Man_data_custom,min_dimension,max_dimension,step) #graficamos 

