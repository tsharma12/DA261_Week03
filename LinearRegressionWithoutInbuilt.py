# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:44:51 2024

@author: teesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

data = pd.DataFrame([[2,3],[4,7],[6,5],[8,10]])
x = data.iloc[:,0]
y = data.iloc[:,1]

plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Building the model
m = 0
c = 0
L = 0.001
epochs = 100
n = len(x)
history = pd.DataFrame()
prev_cost = float('Inf')
stopping_threshold = 0.01
# perform gradient descent
for i in range(epochs):
    print(i)
    Y_pred = m*x + c
    D_m = (-2/n)*sum(x*(y-Y_pred))
    D_c = (-2/n)*sum(y-Y_pred)
    m = m - L*D_m
    c = c - L*D_c
    Y_pred = m*x + c
    slope = round(m,3)
    intercept = round(c,3)
    hist = pd.DataFrame({"Iteration":i,
                         "Slope":slope,
                         "Intercept":intercept,
                         "D_slope":-D_m,
                         "D_intercept":-D_c},index=[i])
    history = pd.concat([history,hist])
    
    current_cost = np.mean((y-Y_pred)**2)
    print(f'Iteration {i} - Slope: {slope}, Intercept: {intercept}, Cost: {current_cost}')
    
    if abs(prev_cost - current_cost) < stopping_threshold:
        print("The stopping criteria is met.")
        break
    
    prev_cost = current_cost
    plt.scatter(x, y)
    plt.plot([min(x),max(x)],[min(Y_pred),max(Y_pred)],
             color="red",
             label=f'y={slope}x+{intercept}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    # time.sleep(1)
    
plt.plot(history['Iteration'],history['Slope'],
         color='red',label='Slope')   
plt.plot(history['Iteration'],history['Intercept'],
         color='green',label='Intercept')
plt.xlabel('No. of Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()

plt.plot(history['Iteration'],history['D_slope'],
         color='red',label='Partial derivative w.r.t. slope')
plt.xlabel('No. of Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()

plt.plot(history['Iteration'],history['D_intercept'],
         color='green',label='Partial derivative w.r.t. intercept')
plt.xlabel('No. of Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()






