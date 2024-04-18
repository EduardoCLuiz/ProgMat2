import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from funcoes import funcao_quadratica, calcula_gradiente, calcula_hessiana_simbolica
import sys

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
        self.pontos_intermediarios = []  # Lista para armazenar os pontos intermediários durante a otimização

    def calcula_funcao(self, ponto_k):
        # Método para calcular o valor da função objetivo em um determinado ponto
        return self.funcao(ponto_k)

    def calcula_gradiente(self, ponto_k):
        # Método para calcular o gradiente da função objetivo em um determinado ponto
        return self.gradiente(self.funcao, ponto_k)

    def calcula_hessiana(self, ponto_k):
        # Método para calcular a hessiana da função objetivo em um determinado ponto
        return self.hessiana(self.funcao, ponto_k)

    def busca_linear_exato(self, ponto_k, direcao):
        # Método para realizar a busca linear exata
        alfa = 0.99
        lambda = 1
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
            self.pontos_intermediarios.append(ponto_k.copy())

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

            # Verifica se a Hessiana é singular
            #if np.linalg.cond(hessiana_fk) < 1/sys.float_info.epsilon:
                #break

            # Resolve o sistema linear para encontrar dk
            dk = np.linalg.solve(hessiana_fk, -gradiente_fk)

            alfa_k = self.busca_linear(ponto_k, dk)

            ponto_k = ponto_k + alfa_k * dk
            self.pontos_intermediarios.append(ponto_k.copy())

            if np.linalg.norm(gradiente_fk) < tol:
                break

            iteracao += 1

        return ponto_k

# Define o ponto inicial
ponto_inicial_exemplo = np.array([3.0, 3.0])

# Cria uma instância da classe de otimização
otimizacao = OtimizacaoSemRestricoes(ponto_inicial_exemplo, funcao_quadratica, calcula_gradiente, calcula_hessiana_simbolica)

# Executa o método de Newton
ponto_otimo_newton = otimizacao.metodo_newton(ponto_inicial_exemplo)

# Executa o método do gradiente descendente
ponto_otimo = otimizacao.metodo_gradiente(ponto_inicial_exemplo)

# Imprime os pontos intermediários do método de Newton
print("Pontos intermediários (Newton):")
for i, ponto_intermediario in enumerate(otimizacao.pontos_intermediarios):
    ponto_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_intermediario]
    print(f"Iteração {i+1}: {ponto_formatado}")

# Formata o ponto ótimo encontrado pelo método de Newton para impressão
ponto_otimo_newton_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_otimo_newton]
print("Ponto ótimo encontrado pelo método de Newton:", ponto_otimo_newton_formatado)
print("Valor ótimo encontrado pelo método de Newton:", round(otimizacao.calcula_funcao(ponto_otimo_newton), 3))

# Imprime os pontos intermediários do método do gradiente descendente
print("Pontos intermediários (Gradiente Descendente):")
for i, ponto_intermediario in enumerate(otimizacao.pontos_intermediarios):
    ponto_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_intermediario]
    print(f"Iteração {i+1}: {ponto_formatado}")

# Formata o ponto ótimo do método do gradiente descendente para impressão
ponto_otimo_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_otimo]
print("Ponto ótimo encontrado pelo método do gradiente:", ponto_otimo_formatado)
print("Valor ótimo encontrado pelo método do gradiente:", round(otimizacao.calcula_funcao(ponto_otimo), 3))

# Extrai os pontos intermediários do método de Newton para plotagem
pontos_x_newton = [p[0] for p in otimizacao.pontos_intermediarios]
pontos_y_newton = [p[1] for p in otimizacao.pontos_intermediarios]
pontos_z_newton = [otimizacao.calcula_funcao(p) for p in otimizacao.pontos_intermediarios]

# Extrai os pontos intermediários do método do gradiente descendente para plotagem
pontos_x_gradiente = [p[0] for p in otimizacao.pontos_intermediarios]
pontos_y_gradiente = [p[1] for p in otimizacao.pontos_intermediarios]
pontos_z_gradiente = [otimizacao.calcula_funcao(p) for p in otimizacao.pontos_intermediarios]

# Plotagem da função e dos pontos intermediários dos dois métodos em um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotagem da superfície da função
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = funcao_quadratica(np.array([X, Y]))
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plotagem dos pontos intermediários do método de Newton com linhas numeradas
for i, (x, y, z) in enumerate(zip(pontos_x_newton, pontos_y_newton, pontos_z_newton), 1):
    ax.text(x, y, z, str(i), color='red', fontsize=10)
    if i < len(pontos_x_newton):
        ax.plot([pontos_x_newton[i-1], pontos_x_newton[i]], [pontos_y_newton[i-1], pontos_y_newton[i]], [pontos_z_newton[i-1], pontos_z_newton[i]], color='blue')

# Plotagem dos pontos intermediários do método do gradiente descendente com linhas numeradas
for i, (x, y, z) in enumerate(zip(pontos_x_gradiente, pontos_y_gradiente, pontos_z_gradiente), 1):
    ax.text(x, y, z, str(i), color='green', fontsize=10)
    if i < len(pontos_x_gradiente):
        ax.plot([pontos_x_gradiente[i-1], pontos_x_gradiente[i]], [pontos_y_gradiente[i-1], pontos_y_gradiente[i]], [pontos_z_gradiente[i-1], pontos_z_gradiente[i]], color='yellow')

# Configurações do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Função Quadrática e Pontos Intermediários dos Métodos de Newton e Gradiente Descendente')
plt.show()





#busca linear (4.2)-> newton
