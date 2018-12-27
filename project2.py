import glob
import os
import numpy as np
import pickle
#from sklearn.model_selection import train_test_split

import importlib
import logisRegresANA



def main():
  np.random.seed(1) # shuffle random seed generator
  """
  # Ising model parameters
  L=40 # linear system size
  J=-1.0 # Ising interaction
  T=np.linspace(0.25,4.0,16) # set of temperatures
  T_c=2.26 # Onsager critical temperature in the TD limit
  ##### prepare training and test data sets
  ###### define ML parameters
  num_classes=2
  train_to_test_ratio=0.5 # training samples

  # path to data directory
  path_to_data=os.path.expanduser('.')+'/data/'

  # load data
  file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
  file = open(path_to_data+file_name,'rb')
  data = pickle.load(file) # pickle reads the file and returns the Python object (1D array, compressed bits)
  data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
  #type(data)
  #data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)
  file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
  labels = pickle.load(open(path_to_data+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)
  # divide data into ordered, critical and disordered
  X_ordered=data[:70000,:]
  Y_ordered=labels[:70000]
  X_critical=data[70000:100000,:]
  Y_critical=labels[70000:100000]

  X_disordered=data[100000:,:]
  Y_disordered=labels[100000:]

  X_ordered[np.where(X_ordered==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)
  X_critical[np.where(X_critical==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)
  X_disordered[np.where(X_disordered==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)
  del data,labels

  # define training and test data sets
  X=np.concatenate((X_ordered,X_disordered))
  Y=np.concatenate((Y_ordered,Y_disordered))

  # pick random data points from ordered and disordered states
  # to create the training and test sets
  test_size = 1. - train_to_test_ratio
  X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size) #train_size=train_to_test_ratio)

  # full data set
  X=np.concatenate((X_critical,X))
  Y=np.concatenate((Y_critical,Y))

  print('X_train shape:', X_train.shape)
  print('Y_train shape:', Y_train.shape)
  print()
  print(X_train.shape[0], 'train samples')
  print(X_critical.shape[0], 'critical samples')
  print(X_test.shape[0], 'test samples')

  file = open("inputNN", 'wb')
  data = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}
  pickle.dump(data, file)
  file.close()
"""
  file = open("inputNN", 'rb')
  data = pickle.load(file)
  file.close()
  X_train = data['X_train']
  Y_train = data['Y_train']
  X_test = data['X_test']
  Y_test = data['Y_test']
  importlib.reload(logisRegresANA)
  in_layer = X_train.shape[1] #number of neurons in the input layer

  if (len(Y_train.shape)==1):
        out_layer = 1   #number of neurons in the output layer
  else: out_layer = Y_train.shape[1]
  biasesnn, weightsnn= logisRegresANA.neuralnetwork([in_layer, 10, out_layer], X_train,
                                                  Y_train,
                                                  validation_x=X_test, validation_y=Y_test,
                                                  verbose=True,
                             epochs= 1, mini_batch_size = 10, lr= 0.5, C='ce')
  print('biasesnn ', biasesnn)
  print('weightsnn ', weightsnn)
  file = open("stateNN", 'wb')
  data = {'biases': biasesnn, 'weights': weightsnn}
  pickle.dump(data, file)
  file.close()

if __name__ == '__main__':
    main()
