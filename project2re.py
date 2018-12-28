import glob
import os
import numpy as np
import pickle
import importlib
import logisRegresANA
import warnings
def ising_energies(states,L):
  """
  This function calculates the energies of the states in the nn Ising Hamiltonian
  """
  J=np.zeros((L,L),)
  for i in range(L):
      J[i,(i+1)%L]-=1.0
  # compute energies
  E = np.einsum('...i,ij,...j->...',states,J,states)

  return E

def main():
  np.random.seed(12)
  warnings.filterwarnings('ignore')

  ### define Ising model aprams
  # system size
  L=40

  # create 10000 random Ising states
  states=np.random.choice([-1, 1], size=(10000,L))

  # calculate Ising energies
  energies=ising_energies(states,L)
  # reshape Ising states into RL samples: S_iS_j --> X_p
  states=np.einsum('...i,...j->...ij', states, states)
  shape=states.shape
  states=states.reshape((shape[0],shape[1]*shape[2]))
  # build final data set
  Data=[states,energies]
  # define number of samples
  n_samples=400
  # define train and test data sets
  X_train=Data[0][:n_samples]
  Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
  X_test=Data[0][n_samples:3*n_samples//2]
  Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

  importlib.reload(logisRegresANA)
  in_layer = X_train.shape[1] #number of neurons in the input layer

  if (len(Y_train.shape)==1):
        out_layer = 1   #number of neurons in the output layer
  else: out_layer = Y_train.shape[1]
  biasesnn, weightsnn= logisRegresANA.neuralnetwork([in_layer, 10, out_layer], X_train,
                                                  Y_train,
                                                  validation_x=X_test, validation_y=Y_test,
                                                  verbose=True,
                             epochs= 1, mini_batch_size = 10,  C='re')
  print('biasesnn ', biasesnn)
  print('weightsnn ', weightsnn)
  file = open("RstateNN", 'wb')
  data = {'biases': biasesnn, 'weights': weightsnn}
  pickle.dump(data, file)
  file.close()

if __name__ == '__main__':
    main()
