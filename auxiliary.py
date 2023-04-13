import numpy as np
import matplotlib.pyplot as plt 

def Plot_data(euclidean_data,manhattan_data,min_dim,max_dim,step_dim,Eu_data,Man_data):
  """
    Parameters
    ----------
    euclidean_data : TYPE
        DESCRIPTION.
    manhattan_data : TYPE
        DESCRIPTION.
    min_dim : TYPE
        DESCRIPTION.
    max_dim : TYPE
        DESCRIPTION.
    step_dim : TYPE
        DESCRIPTION.
    Eu_data : TYPE
        DESCRIPTION.
    Man_data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
  l = len(euclidean_data)
  dimensiones = np.arange(min_dim,max_dim,step_dim)
  eu_R = np.zeros(l)
  eu_CSI = np.zeros(l)
  eu_VSI = np.zeros(l)
  man_R = np.zeros(l)
  man_CSI = np.zeros(l)
  man_VSI = np.zeros(l)
  for i in range(l):
    eu_R[i] = Eu_data[i][0]
    eu_CSI[i] = Eu_data[i][1]
    eu_VSI[i] = Eu_data[i][2]
    man_R[i] = Man_data[i][0]
    man_CSI[i] = Man_data[i][1]
    man_VSI[i] = Man_data[i][2]


  #Graficamos 
  fig, ax = plt.subplots(1, 2, figsize=(16,6), sharey = False)
  fig.tight_layout(pad=5.0) #espaciar las graficas

  ax[0].plot(dimensiones, eu_R, color='r', label='euclidean')
  ax[0].plot(dimensiones, man_R, color='g', label='manhattan')
  ax[0].set_title('R^2: Euclidean vs Manhatta')
  ax[0].legend()


  ax[1].plot(dimensiones, eu_CSI, color='r', label='euclidean')
  ax[1].plot(dimensiones, man_CSI, color='g', label='manhattan')
  ax[1].set_title('CSI: Euclidean vs Manhatta')
  ax[1].legend()

#No tiene sentido graficar el VSI si cogemos todos los atributos disponibles para la explicacion
# ax[1,1].plot(dimensiones, eu_VSI, color='r', label='euclidean')
#  ax[1,1].plot(dimensiones, man_VSI, color='g', label='manhattan')
#  ax[1,1].set_title('VSI: Euclidean vs Manhatta')
#  ax[1,1].legend()

  plt.show()

def Explication_study(instance,num_features,num_samples,distance,explainer,classifier, X_test):
    """
    Parameters
    ----------
    instance : TYPE
        DESCRIPTION.
    num_features : TYPE
        DESCRIPTION.
    num_samples : TYPE
        DESCRIPTION.
    distance : TYPE
        DESCRIPTION.
    explainer : TYPE
        DESCRIPTION.
    classifier : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    csi : TYPE
        DESCRIPTION.
    vsi : TYPE
        DESCRIPTION.

    """
    exp = explainer.explain_instance(X_test[instance],classifier.predict_proba, 
                                   num_features=num_features,num_samples=num_samples,
                                   distance_metric=distance)
  
    csi,vsi = explainer.check_stability(X_test[instance],classifier.predict_proba, 
                                          num_features=num_features,num_samples=num_samples,
                                          distance_metric=distance,n_calls = 10)
    return exp.score,csi,vsi
