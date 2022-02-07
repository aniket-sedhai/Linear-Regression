# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 22:33:57 2022

@author: aniket-sedhai
"""


import numpy as np
from matplotlib import pyplot as plt

#Initializing empty lists for x and y data
x = []
y = []

#Reading data from the txt file and storing it as a matrix of dimension 97x2
data = np.loadtxt('food_truck_data.txt', delimiter = ",", skiprows = 1)
m = len(data)
for i in range(m):
    x.append(data[i][0])                            #Reads the first column from each row and add it to the x list
    y.append(data[i][1])                            #Reads the second column from each row and add it to the y list

#Reshaping list into array of m rows and 1 column
x = np.array(x).reshape(m,1)                      
y = np.array(y).reshape(m,1)                        

#Plotting the first figure which has the scatterplot y vs. x
plot1 = plt.figure(1)
plt.plot(x, y, 'o', color='black')
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Food Truck Data")

#Initializing W which consists of our target values for the linear regression
W = [0,0]       #W[0] = slope of the line      #W[1] = y-intercept

#y_cap stores the value predicted using our equation
y_cap = 0

#Defining the cost function to see how much error we get for each W we generate
def Cost_Function(x, y, W):
    sum_of_errors = 0
    for i in range(len(x)):
        y_cap = x[i]*W[0] + W[1]
        error = y_cap - y[i]
        sum_of_errors += np.power(error, 2)
    Cost = sum_of_errors/(2*m)
    return Cost

#Defining the update function that updates the W to find the best fit
def update(x, y, W, alpha):
    dw0 = 0
    dw1 = 0
    N = len(x)
    
    #Computes the gradient of the cost function wrt W[0] and W[1]
    for i in range (N):
        dw0 += -2*x[i]*(y[i] - (W[0]*x[i] + W[1]))
        dw1 += -2*(y[i] - (W[0]*x[i] + W[1]))
        
    #Update W    
    W[0] = W[0] - 1/float(N)*alpha*dw0
    W[1] = W[1] - 1/float(N)*alpha*dw1
    return W
#This vector will store all our values for the cost function output
myCost = []

#Training function that runs the update function desired number of times and 
#calculates cost for each iteration and stores in the list we created

def train(x, y, W, alpha, iterations):
    for n in range(iterations):
        W = update(x, y, W, alpha)
        Cost = Cost_Function(x, y, W)
        myCost.append(Cost)
    return W


#Let's set the learning rate to be 0.01
alpha = 0.01
W = train(x, y, W, alpha, 5000) #Trains for 5000 times

#Plotting the equation generated
fx = W[0]*x + W[1]
plt.plot(x,fx)

#New plot for the cost function
plot2 = plt.figure(2)
plt.plot(myCost)
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Cost function output")

#Printing the equation we generated that fits the data
myString = "Our linear function fitted to this data is: f(x) = {} x + {}."
print(myString.format(W[1], W[0]))


 


        
    
    
    


