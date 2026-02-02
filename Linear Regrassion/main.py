import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

# y = mx + b

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        
        total_error += (y- (m*x + b))**2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n  = len(points)
    
    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    
    return m, b

m = 0
b = 0
L = 0.0001
epochs = 300
for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    if i % 100 == 0:
        print(f"Epoch {i}: m = {m}, b = {b}, loss = {loss_function(m, b, data)}")

plt.scatter(data.x, data.y, color='black')
plt.plot(data.x, m*data.x + b, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.show()

