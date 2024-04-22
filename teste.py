import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def funcao_quadratica_exemplo(x):
    # Define a função quadrática de exemplo
    return 3 * x[0]**2 + x[1]**2 + 5
    #return 20 * x[0]**2 + x[1] + 20 * x[1]**2

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

class OtimizacaoSemRestricoes:
    def __init__(self, ponto_inicial, funcao, gradiente, hessiana):
        # Inicialização da classe com os parâmetros necessários
        self.ponto_inicial = ponto_inicial  # Ponto inicial da otimização
        self.ponto = ponto_inicial  # Ponto atual da otimização
        self.funcao = funcao  # Função objetivo a ser otimizada
        self.gradiente = gradiente  # Função para calcular o gradiente da função objetivo
        self.hessiana = hessiana  # Função para calcular a hessiana da função objetivo
        self.vetor_gradiente = np.array([1.0, 1.0])  # Vetor gradiente inicial (não utilizado)
        self.matriz_hessiana = None  # Matriz hessiana (não utilizada)
        self.pontos_intermediarios_gradiente = []  # Lista para armazenar os pontos intermediários durante a otimização pelo gradiente
        self.pontos_intermediarios_newton = []  # Lista para armazenar os pontos intermediários durante a otimização pelo método de Newton

    def calcula_funcao(self, ponto_k):
        # Método para calcular o valor da função objetivo em um determinado ponto
        return self.funcao(ponto_k)

    def calcula_gradiente(self, ponto_k):
        # Método para calcular o gradiente da função objetivo em um determinado ponto
        return self.gradiente(self.funcao, ponto_k)

    def calcula_hessiana(self, ponto_k):
        # Método para calcular a hessiana da função objetivo em um determinado ponto
        return self.hessiana(self.funcao, ponto_k)
    
    def busca_linear(self, ponto_k, direcao):
        # Método para realizar a busca linear
        alfa = 1.0
        alpha = 0.5
        max_iter = 100
        reduction_factor = 0.95
        
        f_xk = self.calcula_funcao(ponto_k)
        gradiente_fk = self.calcula_gradiente(ponto_k)
        dk = direcao
        
        for _ in range(max_iter):
            f_xk_novo = self.calcula_funcao(ponto_k + alfa * dk)
            condicao_armijo = f_xk_novo <= f_xk + alpha * alfa * np.dot(gradiente_fk, dk)
            
            if condicao_armijo:
                return alfa
            
            alfa *= reduction_factor
        
        return alfa
    
    def busca_linear_exato(self, ponto_k, direcao):
        # Método para realizar a busca linear exata
        alfa = 1.0
        beta = 0.5
        c = 0.1
        max_iter = 100
        
        f_xk = self.calcula_funcao(ponto_k)
        gradiente_fk = self.calcula_gradiente(ponto_k)
        dk = direcao
        
        for _ in range(max_iter):
            f_xk_novo = self.calcula_funcao(ponto_k + alfa * dk)
            condicao_armijo = f_xk_novo <= f_xk + c * alfa * np.dot(gradiente_fk, dk)
            
            if condicao_armijo:
                return alfa
            
            alfa *= beta
        
        return alfa
    
    def metodo_gradiente(self, ponto_inicial, tol=1e-5, max_iter=1000):
        # Método para otimização usando o gradiente descendente
        ponto_k = ponto_inicial
        iteracao = 0
        
        while iteracao < max_iter:
            gradiente_fk = self.calcula_gradiente(ponto_k)
            dk = -gradiente_fk
            
            if np.allclose(gradiente_fk, 0):
                break
            
            alfa_k = self.busca_linear_exato(ponto_k, dk)
            
            ponto_k = ponto_k + alfa_k * dk
            self.pontos_intermediarios_gradiente.append(ponto_k.copy())
            
            if np.linalg.norm(gradiente_fk) < tol:
                break
            
            iteracao += 1
        
        return ponto_k

    def metodo_newton(self, ponto_inicial, tol=1e-5, max_iter=1000):
        # Método para otimização usando o método de Newton
        ponto_k = ponto_inicial
        iteracao = 0

        while iteracao < max_iter:
            hessiana_fk = self.calcula_hessiana(ponto_k)
            gradiente_fk = self.calcula_gradiente(ponto_k)

            # Resolve o sistema linear para encontrar dk
            dk = np.linalg.solve(hessiana_fk, -gradiente_fk)

            alfa_k = self.busca_linear(ponto_k, dk)

            ponto_k = ponto_k + alfa_k * dk
            self.pontos_intermediarios_newton.append(ponto_k.copy())

            if np.linalg.norm(gradiente_fk) < tol:
                break

            iteracao += 1

        return ponto_k

# Define o ponto inicial
ponto_inicial_exemplo = np.array([3.0, 3.0])

# Cria uma instância da classe de otimização
otimizacao = OtimizacaoSemRestricoes(ponto_inicial_exemplo, funcao_quadratica_exemplo, calcula_gradiente, calcula_hessiana_simbolica)
ponto = np.array([2.0 , 5.0])

# Imprime o gradiente e a hessiana da função no ponto especificado
print("Gradiente da função:", otimizacao.calcula_gradiente(ponto))
print("Hessiana da função:", otimizacao.calcula_hessiana(ponto))

# Executa o método do gradiente descendente
ponto_otimo_gradiente = otimizacao.metodo_gradiente(ponto_inicial_exemplo)

# Imprime os pontos intermediários do método de gradiente
print("Pontos intermediários (Gradiente Descendente):")
for i, ponto_intermediario in enumerate(otimizacao.pontos_intermediarios_gradiente):
    ponto_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_intermediario]
    print(f"Iteração {i+1}: {ponto_formatado}")

# Formata o ponto ótimo do método de gradiente descendente para impressão
ponto_otimo_gradiente_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_otimo_gradiente]
print("Ponto ótimo encontrado pelo método do gradiente:", ponto_otimo_gradiente_formatado)
print("Valor ótimo encontrado pelo método do gradiente:", round(otimizacao.calcula_funcao(ponto_otimo_gradiente), 3))

# Executa o método de Newton
ponto_otimo_newton = otimizacao.metodo_newton(ponto_inicial_exemplo)

# Imprime os pontos intermediários do método de Newton
print("Pontos intermediários (Newton):")
for i, ponto_intermediario in enumerate(otimizacao.pontos_intermediarios_newton):
    ponto_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_intermediario]
    print(f"Iteração {i+1}: {ponto_formatado}")

# Formata o ponto ótimo do método de Newton para impressão
ponto_otimo_newton_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_otimo_newton]
print("Ponto ótimo encontrado pelo método de Newton:", ponto_otimo_newton_formatado)
print("Valor ótimo encontrado pelo método de Newton:", round(otimizacao.calcula_funcao(ponto_otimo_newton), 3))

# Extrai os pontos intermediários para plotagem
pontos_x_gradiente = [p[0] for p in otimizacao.pontos_intermediarios_gradiente]
pontos_y_gradiente = [p[1] for p in otimizacao.pontos_intermediarios_gradiente]
pontos_z_gradiente = [otimizacao.calcula_funcao(p) for p in otimizacao.pontos_intermediarios_gradiente]

pontos_x_newton = [p[0] for p in otimizacao.pontos_intermediarios_newton]
pontos_y_newton = [p[1] for p in otimizacao.pontos_intermediarios_newton]
pontos_z_newton = [otimizacao.calcula_funcao(p) for p in otimizacao.pontos_intermediarios_newton]

# Plotagem do método de gradiente descendente
fig1 = plt.figure()
ax1 = fig1.add_subplot(221, projection='3d')

# Plotagem da superfície da função
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = funcao_quadratica_exemplo(np.array([X, Y]))
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plotagem dos pontos intermediários do método de gradiente descendente
for i, (x, y, z) in enumerate(zip(pontos_x_gradiente, pontos_y_gradiente, pontos_z_gradiente), 1):
    ax1.text(x, y, z, str(i), color='red', fontsize=10)
    if i < len(pontos_x_gradiente):
        ax1.plot([pontos_x_gradiente[i-1], pontos_x_gradiente[i]], [pontos_y_gradiente[i-1], pontos_y_gradiente[i]], [pontos_z_gradiente[i-1], pontos_z_gradiente[i]], color='blue')

# Plotagem do ponto ótimo do método de gradiente em uma cor diferente
ax1.scatter(float(ponto_otimo_gradiente_formatado[0]), float(ponto_otimo_gradiente_formatado[1]), otimizacao.calcula_funcao(ponto_otimo_gradiente), color='green', s=100, label='Gradiente Descendente')

# Configurações do gráfico
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(X, Y)')
ax1.set_title('Método do Gradiente Descendente')
ax1.legend()
######
# Plotagem do método de Newton
ax2 = fig1.add_subplot(222, projection='3d')

# Plotagem da superfície da função
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plotagem dos pontos intermediários do método de Newton
for i, (x, y, z) in enumerate(zip(pontos_x_newton, pontos_y_newton, pontos_z_newton), 1):
    ax2.text(x, y, z, str(i), color='red', fontsize=10)
    if i < len(pontos_x_newton):
        ax2.plot([pontos_x_newton[i-1], pontos_x_newton[i]], [pontos_y_newton[i-1], pontos_y_newton[i]], [pontos_z_newton[i-1], pontos_z_newton[i]], color='blue')

# Plotagem do ponto ótimo do método de Newton em uma cor diferente
ax2.scatter(float(ponto_otimo_newton_formatado[0]), float(ponto_otimo_newton_formatado[1]), otimizacao.calcula_funcao(ponto_otimo_newton), color='red', s=100, label='Newton')

# Configurações do gráfico
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('f(X, Y)')
ax2.set_title('Método de Newton')
ax2.legend()

# Plotagem das curvas de nível
ax3 = fig1.add_subplot(223)
ax3.contour(X, Y, Z, 20, cmap='viridis')

# Plotagem dos pontos intermediários do método de gradiente descendente
ax3.plot(pontos_x_gradiente, pontos_y_gradiente, color='blue', label='Gradiente Descendente')
ax3.scatter(pontos_x_gradiente, pontos_y_gradiente, color='red')

# Plotagem do ponto ótimo do método de gradiente em uma cor diferente
ax3.scatter(float(ponto_otimo_gradiente_formatado[0]), float(ponto_otimo_gradiente_formatado[1]), color='green', s=100, label='Gradiente Descendente - Ótimo')

# Configurações do gráfico
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Curvas de Nível com Método do Gradiente Descendente')
ax3.legend()

# Plotagem das curvas de nível
ax4 = fig1.add_subplot(224)
ax4.contour(X, Y, Z, 20, cmap='viridis')

# Plotagem dos pontos intermediários do método de Newton
ax4.plot(pontos_x_newton, pontos_y_newton, color='red', label='Newton')
ax4.scatter(pontos_x_newton, pontos_y_newton, color='blue')

# Plotagem do ponto ótimo do método de Newton em uma cor diferente
ax4.scatter(float(ponto_otimo_newton_formatado[0]), float(ponto_otimo_newton_formatado[1]), color='red', s=100, label='Newton - Ótimo')

# Configurações do gráfico
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('Curvas de Nível com Método de Newton')
ax4.legend()

# Ajusta o layout e mostra a figura
plt.tight_layout()
plt.show()
