import numpy as np
import sympy as sp

def funcao_quadratica(x):
    # Define a função quadrática
    return 20 * x[0]**2 + 1 * x[1] + 20 * x[0]**2 + x[1]**2
    #return 3 * x[0]**2 + x[1]**2 + 5

def calcula_gradiente(funcao, x):
    # Calcula o gradiente da função numericamente
    h = 1e-5
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += h
        x2[i] -= h
        grad[i] = (funcao(x1) - funcao(x2)) / (2 * h)
    return grad

def calcula_hessiana_simbolica(funcao, x):
    # Calcula a hessiana da função simbolicamente usando SymPy
    x_sympy = [sp.symbols('x'+str(i)) for i in range(len(x))]  # Define símbolos para cada variável
    f = funcao(x_sympy)  # Cria a expressão da função usando os símbolos
    hessiana = sp.hessian(f, x_sympy)  # Calcula a matriz hessiana da função
    hessiana_eval = hessiana.subs([(x_sympy[i], x[i]) for i in range(len(x))])  # Avalia a hessiana nos pontos fornecidos
    return np.array(hessiana_eval).astype(float)  # Converte a matriz hessiana para um array numpy




#vufrrj