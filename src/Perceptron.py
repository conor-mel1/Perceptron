'''
@author: Conor Melody

'''
import numpy as np
import math
import sys
from matplotlib import pyplot as plt


class CPerceptron :
    """This is is the ML algorithmn we will use to predict the type of owl. Like in sci-kit learn, the different 
    methods/functions associated with the ML algorithm are all grouped together in a single class.
    Using the training data, and the 'fit_weights' function the weights associated with each parameter will be 
    calculated, (ie the model will be trained).
    Then, having trained the model, we will be able to make predictions on unseen data using the 
    the 'predict_sigmoid' function. 
    The accuracy of these predictions will then be evaluated using the "Accuracy_metric" function """
 
 
    """To initialise the class, an init or constructor method must first be defined.
    The parameter 'number_of_weights' refers to the number of weights needed on each of the
     input parameters in the particular problem. In this particular problem, we have 4 weights associated with each parameter
     and a fifth weight which is the bias.
     A higher weighting on any particular parameter indicates that this parameter has a greater influence on the 
     predicted result than other parameters."""
    def __init__(self, number_of_weights):
        self.weights = [0.0 for i in range(number_of_weights)]
        self.number_of_weights = number_of_weights
        
        
    """ Before training the model, we cannot assume any of the parameters are better predictors (ie have a higher
    weight associated with them) than the others. Therefore the initial weights are all set to be zero using this function.
    Gradient descent will then be used to determine appropriate weights for each parameter."""
    def initialise_weights(self):
        self.weights = [0.0 for i in range(len(self.weights))]
    


    """ This fit_weights function calculates the weights associated with the data using stochastic gradient descent.
    Given a learning rate (alpha), a set of training data, and iterating the algorithm a certain number of times, 
    (bearing in mind the need to decay alpha in proportion to 1/t where t is the number of iterations), the weights
    for each parameter are calculated.""" 
    def fit_weights(self, X_train, learning_rate, n_iterations):
        for iteration in range(n_iterations):
            for row in X_train:
                prediction = self.predict_sigmoid(row)
                error = row[-1] - prediction #last input of row of data in X_train, ie 1 or 0. Prediction is either
                # 1 or 0 from Sigmoid function. Hence the error is either 1 or 0. 
                self.weights[0] = self.weights[0] + learning_rate * error # This is updating the bias (weight 0)
                for i in range(len(row)-1):
                    self.weights[i + 1] = self.weights[i + 1] + learning_rate * error * row[i]
        return
 
    """ This function is used to calculate predictions from test data. A soft threshold is applied using the 
    sigmoid function (f(x)=1/1+e^-x). This function will output a probability between 0 and 1. 
    If the output probability prob_y is greater than or equal to 0.5, the prediction function will return a 1."""
    def predict_sigmoid(self, X_test_row):
        bias = self.weights[0] 
        for i in range(len(X_test_row)-1):
            dot_product = bias + self.weights[i + 1] * X_test_row[i]
            prob_y =  1 / (1 + math.exp(-dot_product))
            #print("Prob_y", prob_y)
        return 1.0 if prob_y >= 0.5 else 0.0   
 
 
 
    """This function creates a list of the predicted outcomes using the sigmoid prediction function."""
    def predict(self, test_data): 
        predictions = list()
        for row in test_data:
            prediction = self.predict_sigmoid(row)
            predictions.append(prediction)
        return(predictions)    
        
    """ The measure_accuracy function is how the accuracy of the ML algorithm is evaluated. It simply calculates the
    number of correctly predicted outcomes from the test data and uses this number to give a percentage accuracy
    of the algorithm. If the predicted output is 1 and the actual output is 0, this is a false positive. 
    Similarly, if the predicted output is 0 and the actual output is 1, this is a false negative """
    def measure_accuracy(self, actual, predicted, output_file):
        correct=0
        false_positives=0
        false_negatives=0
        for i in range(len(actual)):
            error = actual[i] - predicted[i]
            if error == 0:
                print("Actual = ", actual[i], "Predicted =", predicted[i], "=> CORRECT", file=open(output_file, "a") )
                correct += 1
            elif error < 0:
                false_positives += 1
                print("Actual = ", actual[i], "Predicted =", predicted[i], "=> INCORRECT (False Positive)", file=open(output_file, "a") )
            elif error > 0:
                false_negatives += 1
                print("Actual = ", actual[i], "Predicted =", predicted[i], "=> INCORRECT (False Negative)", file=open(output_file, "a") )
                correct += 1
        return correct / float(len(actual)) *100.0

#Run the program by typing python MLss3.py <input filename> <output filename> <number of features>
if len(sys.argv) != 4:
    print("usage:python MLss3.py <input filename> <output filename> <number of features> ")
    sys.exit(0)
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    number_of_features = int(sys.argv[3])

dataset1 = np.loadtxt(input_file, delimiter = ",")
    

learning_rate = 0.01
n_iterations = 100

scores = list()

for i in range(10):
#Shuffling dataset so as to create different test and training subsets for each loop.
    np.random.shuffle(dataset1)

#Selecting training and testing datasets.
    myList = list(range(0,len(dataset1)))

#Selecting rows that are not a multiple of 3, ie two thirds of the data for training
    selected_train_rows = [x for x in myList if x % 3 != 0]
    X_train = dataset1[selected_train_rows]
    #print("X_train", X_train)
    #print("X_train shape", X_train.shape)

#Selecting rows that are a multiple of 3, ie one third of the data
    selected_test_rows= [x for x in myList if x % 3 == 0]
    X_test = dataset1[selected_test_rows]  
    #print("X_test", X_test)
    #print("X_test shape", X_test.shape)

    # The number of weights is 5 (the number of features plus the bias)
    # as we have a weight associated with each parameter plus a bias
    perCept= CPerceptron(number_of_features + 1 )
    perCept.initialise_weights()

    #Fitting the weights for each parameter
    perCept.fit_weights(X_train, learning_rate, n_iterations)
    print(perCept.weights)
    
    #Using test data to generate predictions
    predicted = perCept.predict(X_test)

    #Selecting last column of X_test as first 4 columns are parameters and last column is outcome.
    #These outcomes are then compared against the predicted outcomes, testing the accuracy of the predictions.
    accuracy = perCept.measure_accuracy(X_test[:,-1],predicted, output_file )
    
    print("Accuracy", accuracy)
    scores.append(accuracy)
    print(" ")
    print('=============== Loop-number = %d ==================' % i)
    print(" ")
    
print("Scores",scores, file=open("Scores.txt", "a"))
print("Mean Accuracy", (sum(scores)/float(len(scores))))

# plot the cross validation scores 
x_list = range(1,11,1)
mean_accuracy = sum(scores)/float(len(scores))
# generate an array of the means to show on the plot 
score_means = [mean_accuracy]*len(x_list)


# Setup the plots 
fig,ax = plt.subplots()
#
data_line1 = ax.plot(x_list, scores, label=' scores', marker='o')
data_line2 = ax.plot(x_list, score_means, label=' mean score', linestyle='-')

# PLot legend 
legend = ax.legend(loc='lower right')
plt.ylim((0, 102)) 
plt.title(input_file)

plt.xlabel("Loop Number")
plt.ylabel("Percentage Accuracy")
plt.show()

    


  


