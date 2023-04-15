import numpy as np
import matplotlib.pyplot as plt 


def Plot_data(euclidean_data,manhattan_data,manhattan_data_custom,min_dim=5,max_dim=40,step_dim=5):
  """
    Parameters
    ----------
    euclidean_data : 
        
    manhattan_data : 
        
    min_dim : Int
        Minimum dimensionality for the data set generation.
    max_dim : Int
        Maximum dimensionality for the data set generation.
    step_dim : Int
        Difference on dimensionality between data sets.

    Returns

    Graphics of each measure R^2, CSI, VSI and similarity over dimensionality.

    """
  
  dimensiones = np.arange(min_dim,max_dim,step_dim)
  l = len(dimensiones)
  eu_R = np.zeros(l)
  eu_CSI = np.zeros(l)

  man_R = np.zeros(l)
  man_CSI = np.zeros(l)

  man_R_custom = np.zeros(l)
  man_CSI_custom = np.zeros(l)
  
  for i in range(l):
    eu_R[i] = np.mean(euclidean_data[0][i])
    eu_CSI[i] = np.mean(euclidean_data[1][i])

    man_R[i] = np.mean(manhattan_data[0][i])
    man_CSI[i] = np.mean(manhattan_data[1][i])

    man_R_custom[i] = np.mean(manhattan_data_custom[0][i])
    man_CSI_custom[i] = np.mean(manhattan_data_custom[1][i])
   
  


  #Graficamos 
  fig, ax = plt.subplots(2, 2, figsize=(16,6), sharey = False)
  fig.tight_layout(pad=5.0) #espaciar las graficas

  ax[0,0].plot(dimensiones, eu_R, color='r', label='euclidean')
  ax[0,0].plot(dimensiones, man_R, color='g', label='manhattan')
  ax[0,0].plot(dimensiones,man_R_custom, color='b',label='manhattan_custom_kw')
  ax[0,0].set_title('R^2: Euclidean vs Manhattan')
  ax[0,0].legend()

  ax[0,1].plot(dimensiones, eu_CSI, color='r', label='euclidean')
  ax[0,1].plot(dimensiones, man_CSI, color='g', label='manhattan')
  ax[0,1].plot(dimensiones,man_CSI_custom, color='b',label='manhattan_custom_kw')
  ax[0,1].set_title('CSI: Euclidean vs Manhattan')
  ax[0,1].legend()

  eu_r_boxplot=ax[1,0].boxplot(euclidean_data[0],patch_artist=True)
  man_r_boxplot=ax[1,0].boxplot(manhattan_data[0],patch_artist=True)
  man_r_custom_boxplot=ax[1,0].boxplot(manhattan_data_custom[0],patch_artist=True)
  

  #La coloracion de los boxplot esta mal 
  colors = ['red', 'green', 'blue']
  for bplot in (eu_r_boxplot, man_r_boxplot, man_r_custom_boxplot):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

  eu_csi_boxplot=ax[1,1].boxplot(euclidean_data[1],patch_artist=True)
  man_csi_boxplot=ax[1,1].boxplot(manhattan_data[1],patch_artist=True)
  man_csi_custom_boxplot=ax[1,1].boxplot(manhattan_data_custom[1],patch_artist=True)
  
  colors = ['red', 'green', 'blue']
  for bplot in (eu_csi_boxplot, man_csi_boxplot, man_csi_custom_boxplot):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
  
  
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

