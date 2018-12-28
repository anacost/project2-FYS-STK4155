import glob
import os
import numpy as np
import pickle
import importlib
import logisRegresANA

def main():
  np.random.seed(1) # shuffle random seed generator

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
                             epochs= 1, mini_batch_size = 10,  C='ce')
  print('biasesnn ', biasesnn)
  print('weightsnn ', weightsnn)
  file = open("stateNN", 'wb')
  data = {'biases': biasesnn, 'weights': weightsnn}
  pickle.dump(data, file)
  file.close()
 
if __name__ == '__main__':
    main()
