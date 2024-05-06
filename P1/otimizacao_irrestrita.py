import numpy as np  # Importa a biblioteca NumPy para operações numéricas
import matplotlib.pyplot as plt  # Importa a biblioteca Matplotlib para plotagem de gráficos
from mpl_toolkits.mplot3d import Axes3D  # Importa Axes3D para plotagem de gráficos 3D
from funcoes import funcao, calcula_gradiente, calcula_hessiana  # Importa funções específicas de outro arquivo

# Definição da classe para otimização sem restrições
class OtimizacaoSemRestricoes:
    # Método de inicialização da classe
    def __init__(self, ponto_inicial, funcao, gradiente, hessiana):
        # Inicializa os parâmetros da classe com os valores fornecidos
        self.ponto_inicial = ponto_inicial
        self.ponto = ponto_inicial
        self.funcao = funcao
        self.gradiente = gradiente
        self.hessiana = hessiana
        self.pontos_intermediarios_gradiente = []  # Lista para armazenar os pontos intermediários do gradiente
        self.pontos_intermediarios_newton = []  # Lista para armazenar os pontos intermediários do método de Newton
        self.pontos_intermediarios_descida = []  # Lista para armazenar os pontos intermediários do método de descida

    # Método para calcular o valor da função objetivo em um determinado ponto
    def calcula_funcao(self, ponto_k):
        return self.funcao(ponto_k)

    # Método para otimização usando o método de descida
    def metodo_descida(self, ponto_inicial, a=0.5, tol=1e-5, max_iter=1000):
        ponto_k = ponto_inicial  # Define o ponto inicial
        iteracao = 0  # Inicializa a contagem de iterações

        # Matriz Z cujas colunas formam uma base do núcleo de A
        Z = np.eye(len(ponto_inicial))[:, len(ponto_inicial):]

        # Loop principal do método de descida
        while iteracao < max_iter:
            gradiente_fk = self.gradiente(ponto_k)  # Calcula o gradiente da função no ponto atual
            Zt_gradiente_fk = np.dot(Z.T, gradiente_fk)  # Multiplica o gradiente pelo transpose de Z

            # Verifica se Z^T * gradiente é igual a zero
            if np.allclose(Zt_gradiente_fk, 0):
                break  # Se for, encerra o loop pois estamos em um ponto estacionário

            # Calcula a direção de descida
            direcao_descida = -np.dot(Z, Zt_gradiente_fk)

            # Realiza a busca linear
            alfa_k = self.busca_linear(ponto_k, direcao_descida, a)

            # Atualiza o ponto utilizando o passo adequado
            ponto_k = ponto_k + alfa_k * direcao_descida
            self.pontos_intermediarios_descida.append(ponto_k.copy())  # Armazena o ponto intermediário

            # Verifica se a norma do gradiente é menor que a tolerância
            if np.linalg.norm(gradiente_fk) < tol:
                break  # Se for, encerra o loop

            iteracao += 1  # Incrementa o número de iterações

        return ponto_k  # Retorna o ponto ótimo encontrado

    # Método para realizar a busca linear
    def busca_linear(self, ponto_k, direcao, a):
        alfa = 1.0  # Inicializa o fator de passo
        alpha = 0.5  # Fator de redução
        max_iter = 100  # Número máximo de iterações
        reduction_factor = 0.95  # Fator de redução

        f_xk = self.calcula_funcao(ponto_k)  # Calcula o valor da função no ponto atual
        gradiente_fk = self.gradiente(ponto_k)  # Calcula o gradiente da função no ponto atual
        dk = direcao  # Direção de descida

        # Loop para encontrar o fator de passo
        for _ in range(max_iter):
            f_xk_novo = self.calcula_funcao(ponto_k + alfa * dk)  # Calcula o valor da função no novo ponto
            condicao_armijo = f_xk_novo <= f_xk + a * alfa * np.dot(gradiente_fk, dk)  # Verifica a condição de Armijo

            if condicao_armijo:
                return alfa  # Se a condição for satisfeita, retorna o fator de passo

            alfa *= reduction_factor  # Reduz o fator de passo

        return alfa  # Retorna o fator de passo

# Define o ponto inicial
ponto_inicial_exemplo = np.array([3.0, -3.0])

# Cria uma instância da classe de otimização
otimizacao = OtimizacaoSemRestricoes(ponto_inicial_exemplo, funcao, calcula_gradiente, calcula_hessiana)

# Imprime o gradiente e a hessiana da função no ponto especificado
print("Gradiente da função:", otimizacao.gradiente(ponto_inicial_exemplo))
print("Hessiana da função:", otimizacao.hessiana(ponto_inicial_exemplo))

# Executa o método de descida
ponto_otimo_descida = otimizacao.metodo_descida(ponto_inicial_exemplo)

# Imprime os pontos intermediários do método de descida
print("Pontos intermediários (Método de Descida):")
for i, ponto_intermediario in enumerate(otimizacao.pontos_intermediarios_descida):
    ponto_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_intermediario]
    print(f"Iteração {i+1}: {ponto_formatado}")

# Formata o ponto ótimo do método de descida para impressão
ponto_otimo_descida_formatado = ['{:.10f}'.format(coord) if abs(coord) > 1e-10 else '0.0' for coord in ponto_otimo_descida]
print("Ponto ótimo encontrado pelo Método de Descida:", ponto_otimo_descida_formatado)
print("Valor ótimo encontrado pelo Método de Descida:", round(otimizacao.calcula_funcao(ponto_otimo_descida), 3))
