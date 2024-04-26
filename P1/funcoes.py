import numpy as np

def funcao(x):
    # Define a função quadrática
    #return 3 * x[0]**2 + x[1]**2 + 5
    # Define a função
    return x[0]**4 + 2*x[0]*x[1] + x[1]**2

def calcula_gradiente(x):
    #return np.array([6*x[0], 2*x[1]])   # Define o gradiente da função quadrática
    return np.array([(4*x[0]**3 + 2*x[1]) , (2*x[0] + 2*x[1])])   # Define o gradiente da função 

def calcula_hessiana(x):
    #return np.array([[6, 0], [0, 2]])   # Define a hessiana da função quadrática
    return np.array([[12*x[0]**2, 2], [2, 2]])   # Define a hessiana da função 
