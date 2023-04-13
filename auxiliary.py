import numpy as np
import matplotlib.pyplot as plt 


def Plot_data(euclidean_data,manhattan_data,similarity_data,min_dim,max_dim,step_dim):
  """
    Parameters
    ----------
    euclidean_data : List of 3 dimensional arrays
        euclidean_data[i] stores the mean R^2, CSI, VSI over all the test set for dimensionality = min_dim+step*i using Euclidean distance.
    manhattan_data : List of 3 dimensional arrays
        manhattan_data[i] stores the mean R^2, CSI, VSI over all the test set for dimensionality = min_dim+step*i using Manhattan distance.
    similarity_data: float list
        similarity_data[i] stores the mean similarity measure over all the test set for dimensionality = min_dim+step*i.
    min_dim : Int
        Minimum dimensionality for the data set generation.
    max_dim : Int
        Maximum dimensionality for the data set generation.
    step_dim : Int
        Difference on dimensionality between data sets.

    Returns

    Graphics of each measure R^2, CSI, VSI and similarity over dimensionality.

    """
  l = len(euclidean_data)
  dimensiones = np.arange(min_dim,max_dim,step_dim)
  eu_R = np.zeros(l)
  eu_CSI = np.zeros(l)
  eu_VSI = np.zeros(l)
  man_R = np.zeros(l)
  man_CSI = np.zeros(l)
  man_VSI = np.zeros(l)
  similitud = np.zeros(l)
  for i in range(l):
    eu_R[i] = euclidean_data[i][0]
    eu_CSI[i] = euclidean_data[i][1]
    eu_VSI[i] = euclidean_data[i][2]
    man_R[i] = manhattan_data[i][0]
    man_CSI[i] = manhattan_data[i][1]
    man_VSI[i] = manhattan_data[i][2]
    similitud[i] = similarity_data[i]


  #Graficamos 
  fig, ax = plt.subplots(2, 2, figsize=(16,6), sharey = False)
  fig.tight_layout(pad=5.0) #espaciar las graficas

  ax[0,0].plot(dimensiones, eu_R, color='r', label='euclidean')
  ax[0,0].plot(dimensiones, man_R, color='g', label='manhattan')
  ax[0,0].set_title('R^2: Euclidean vs Manhattan')
  ax[0,0].legend()

  ax[0,1].plot(dimensiones, eu_CSI, color='r', label='euclidean')
  ax[0,1].plot(dimensiones, man_CSI, color='g', label='manhattan')
  ax[0,1].set_title('CSI: Euclidean vs Manhattan')
  ax[0,1].legend()

  ax[1,0].plot(dimensiones, eu_VSI, color='r', label='euclidean')
  ax[1,0].plot(dimensiones, man_VSI, color='g', label='manhattan')
  ax[1,0].set_title('VSI: Euclidean vs Manhattan')
  ax[1,0].legend()

  ax[1,1].plot(dimensiones, similitud, color='b')
  ax[1,1].set_title('Similarity Euclidean-Manhattan')
  ax[1,1].legend()

  plt.show()


def Explication_study(expl,instance,num_features,num_samples,distance,classifier, X_test):
    """
    Parameters
    ----------
    expl: LimeTabularExplainerOvr
        Class used to explain an instance. Original from LIME, and extended in Optilime with CSI and VSI calculation methods. 
    instance : Int
        Index of the instance in the test set we want to study.
    num_features : Int
        Number of feautres used for the explanation.
    num_samples : Int
        Number of perturbations to train the local linear model.
    distance : String
        Distance to use in LIME framework, "euclidean" or "manhattan".
    classifier : ¿ensemble.RandomForestClassifier?
        Complex model we want to explain.
    X_test : Array
        Array containing the instances of the test set.

    Returns
    -------
    R^2 : Float
        R^2 statistic obtained by the linear model trained to explain X_test[instance]
    csi : Float
        CSI score to measure coefficient stability when explaining X_test[instance] (debería citar optilime aquí?)
    vsi : Float
        VSI score to measure attribute selection stability when explaining X_test[instance]
    used_features: Array
        Contains the attributes selected to explain X_test[instance]

    """
    exp = expl.explain_instance(X_test[instance],classifier.predict_proba, 
                                      num_features=num_features,num_samples=num_samples,
                                      distance_metric=distance)
    used_features = list(zip(*exp.local_exp[1]))[0]
  
    csi,vsi = expl.check_stability(X_test[instance],classifier.predict_proba, 
                                          num_features=num_features,num_samples=num_samples,
                                          distance_metric=distance,n_calls = 10)
    return exp.score,csi,vsi,used_features


def get_S(used_features_eu,used_features_man,dimension):
  """
    Parameters
    ----------
    used_features_eu: Array
        Contains the attributes used for the explanation when using Euclidean distance.
    used_features_man: Array
        Contains the attributes used for the explanation when using Manhattan distance.
    dimension: Int
        Data set dimension. 

    Returns
    -------
    S : Float
        Similarity measure. 

    """
  count = 0
  for f in used_features_eu:
    if f in used_features_man:
      count = count+1
  S = count*2/dimension
  return S