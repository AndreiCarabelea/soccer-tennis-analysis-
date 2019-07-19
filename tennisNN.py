
import pickle
import os
import sys
import numpy as np
from math import exp, log


#data is 2d array or
def truncate(data, maxSize):
    desiredSize = min(np.shape(data)[0], maxSize);
    if data.ndim > 1:
        return data[:desiredSize, :];
    return  data[:desiredSize];




def nnWithTF(train_dataset, train_labels, valid_dataset,
             valid_labels, desiredNumberOfValidationExamples, batch_size):

    
    numTrainingExamples = np.shape(train_labels)[0];

    # the size of validation data set is the same as the size of the training set 
    valid_dataset = truncate(valid_dataset, desiredNumberOfValidationExamples);
    valid_labels = truncate(valid_labels, desiredNumberOfValidationExamples);
    
    num_labels = 1;  
    inputSize = 3;

    graph = tf.Graph();    
    
    with graph.as_default():
    
        tf_train_dataset = tf.placeholder(shape=(batch_size, inputSize), dtype = tf.float32);
        tf_train_labels = tf.placeholder(shape=(batch_size, num_labels), dtype = tf.float32);

        #used for compute crossEntropy error in learning
        tf_valid_dataset = tf.constant(valid_dataset, dtype = tf.float32);
        tf_valid_labels = tf.constant(valid_labels, dtype = tf.float32);

 
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        
        hiddenLayerWidth =  4
        weights1 = tf.Variable(tf.truncated_normal([inputSize, hiddenLayerWidth]), dtype = tf.float32);
        weights2 = tf.Variable(tf.truncated_normal([hiddenLayerWidth, hiddenLayerWidth]), dtype = tf.float32);
        weights3 = tf.Variable(tf.truncated_normal([hiddenLayerWidth, 1]), dtype = tf.float32);

        biases1 = tf.Variable(tf.zeros([hiddenLayerWidth]), dtype = tf.float32);
        biases2 = tf.Variable(tf.zeros([hiddenLayerWidth]), dtype = tf.float32);
        biases3 = tf.Variable(tf.zeros([1]), dtype = tf.float32);

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        # input is a tf tensor
        
        def model(data):
            y1 = tf.matmul(data, weights1) + biases1
            y1 = tf.nn.leaky_relu(y1);
            tf.nn.dropout(y1, 0.8);

            y2 = tf.matmul(y1, weights2) + biases2;
            y2 = tf.nn.leaky_relu(y2);
            y2 = tf.nn.dropout(y2, 0.8);
            #rely on broadcasting
            #result is between 0 and 1 
            return  tf.nn.sigmoid(tf.matmul(y2, weights3) + biases3);
        
        def modelNoDropOut(data):
            y1 = tf.matmul(data, weights1) + biases1
            y1 = tf.nn.leaky_relu(y1);
            
            y2 = tf.matmul(y1, weights2) + biases2;
            y2 = tf.nn.leaky_relu(y2);
            #rely on broadcasting
            #result is between 0 and 1 
            return  tf.nn.sigmoid(tf.matmul(y2, weights3) + biases3);

        
        y3 = model(tf_train_dataset)
        lossTraining = tf.losses.mean_squared_error(tf_train_labels, y3)

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        # learning_rate = tf.train.exponential_decay(learning_rate = 0.6, global_step = global_step, decay_steps=maxNumSteps,decay_rate=0.999, staircase=True);
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step);
        optimizer = tf.train.AdamOptimizer(0.01).minimize(lossTraining);
        
       
        tf_ValidPrediction = modelNoDropOut(tf_valid_dataset);
        lossValidation = tf.losses.mean_squared_error(tf_valid_labels, tf_ValidPrediction)
                
                


    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.

        tf.global_variables_initializer().run();

        lastLossT, lastLossV = 1, 1
        maxNumSteps = 1000 * 1000;
        
       
        
        
        
        for global_step in range(maxNumSteps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.


            offset = global_step * batch_size;
            if( offset > (numTrainingExamples -  batch_size)):
                offset = 0;


            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            
            #fd is feed dictionary
            fd = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

            _, lossT, lossV = session.run([optimizer, lossTraining, lossValidation], feed_dict=fd);
            
            print("Iteration %d" % global_step)

            
            if global_step % 10 == 1:
            
                print("Iteration %d, lossT %.4f, lossV %.4f" % (global_step, lossT, lossV))
                lastLossT = lossT
                lastLossV = lossV
            
                vin = input("Abort training ? ")
                if vin == 'Y':
                    return session.run([weights1, biases1, weights2, biases2, weights3, biases3]) 

#inputData is 4 float array 
#tuple is nn parameters                    
def npModel(tuple, inputData):
    
    #unpack the nn hyper parameters
    weights1, biases1, weights2, biases2, weights3, biases3 = tuple 
    
    y1 = inputData @ weights1 + biases1
    y1 = np.maximum(y1, 0.2*y1);
           

    y2 = y1 @ weights2 + biases2;
    y2 = np.maximum(y2, 0.2*y2);
    
    
    y3 = y2 @ weights3 + biases3
    
    return 1/(1 + np.exp(-y3))           
             
def aggregation(glicko, rank, chain):
     params = [1.89895315,-0.19173725,-1.12081576,0.28512086,0.41437482,0.00829605,0.22179212,0.01510107,0.00039107,1.51561130,-0.34289510,2.46927813, -0.58646244,-1.99290450,-1.06328265,1.54646928]     
     nr = (params[0]*glicko*rank*chain + params[1]*glicko*rank + params[2]*glicko*chain + params[3]*rank*chain + params[4]*glicko + params[5]*rank + params[6]*chain + params[7])
     nm = (params[8]*glicko*rank*chain + params[9]*glicko*rank + params[10]*glicko*chain + params[11]*rank*chain + params[12]*glicko + params[13]*rank + params[14]*chain + params[15])
     return (nr/nm)
    
   
   
if os.path.exists("D:\\pariu\\tennisNN.pickle"):
    tuple = pickle.load(open("D:\\pariu\\tennisNN.pickle", "rb"))
    
else:
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import tensorflow as tf
    
    
    data = pd.read_csv("D:\\pariu\\proiectPariu\\statsReduced.csv")
    print(data.shape)
    
    x = data.loc[:, ['glicko', 'rank', 'chain']]
    y = data.loc[:, ['result']]
    
    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size = 0.3)
    
    npxtrain = xtrain.values
    npytrain = ytrain.values
    
    npxvalid = xvalid.values
    npyvalid = yvalid.values
    
    tuple = nnWithTF(npxtrain, npytrain, npxvalid, npyvalid, 28000, 500)
    
    
    f = open("D:\\pariu\\tennisNN.pickle","wb")
    pickle.dump(tuple, f)
    f.close()
    
#weights1, biases1, weights2, biases2, weights3, biases3 = tuple

while True:
    glickoProb = float(input("Glicko prob:\t"))
    chainProb = float(input("Chain prob:\t"))
    rank1 = float(input("Rank1:\t"))
    rank2 = float(input("Rank2:\t"))
    homeCoef1 = float(input("Home coef 1:\t"))
    homeCoef2 = float(input("Home coef 2:\t"))
    
    
    prank = (log(rank2,2) - log(rank1,2))*0.398
    rankProb = round(exp(prank)/(1+exp(prank)),2)
    
   
    #serveProb = float(input("Serve prob:\t"))
    
    surfaceIndexPlayer1 = float(input("surfaceIndexPlayer1 :\t"))
    surfaceIndexPlayer2 = float(input("surfaceIndexPlayer2:\t"))
    
    
    #chainProb = (chainProb*serveProb)**0.5   
    inputData = np.array([glickoProb, rankProb, chainProb])
    
    fr1 = npModel(tuple, inputData)
    fr1 =  (fr1 * surfaceIndexPlayer1 * homeCoef1) / (fr1 * surfaceIndexPlayer1 * homeCoef1 + (1 - fr1) * surfaceIndexPlayer2 * homeCoef2)
    
    
    inputData2 = 1 - inputData
    fr2 = npModel(tuple, inputData2)
    fr2 =  (fr2 * surfaceIndexPlayer2 * homeCoef2) / (fr2 * surfaceIndexPlayer2 * homeCoef2 + (1 - fr2) * surfaceIndexPlayer1 * homeCoef1)
    
    res = fr1/(fr1 + fr2)
    print("Neural network estimation is %.2f" % res)
    
    res = aggregation(glickoProb, rankProb, chainProb)
    res = (res * surfaceIndexPlayer1 * homeCoef1) / (res * surfaceIndexPlayer1 * homeCoef1 + (1 - res) * surfaceIndexPlayer2 * homeCoef2)
    print("rational  quadratic regression estimation is %.2f" % res)
    
    
    a = input("Abort(Y/N) ?: ")
    if a == "Y" or a == "y":
        sys.exit()



