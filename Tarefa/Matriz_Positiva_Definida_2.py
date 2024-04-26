import numpy as np

def determinante_matriz(matriz):
    # Verificar se a matriz é quadrada
    if matriz.shape[0] != matriz.shape[1]:
        return None

    # Caso base: matriz 1x1
    if matriz.shape == (1, 1):
        return matriz[0, 0]

    det_total = 0

    # Calcular o determinante recursivamente
    for coluna in range(matriz.shape[1]):
        menor = np.delete(np.delete(matriz, 0, axis=0), coluna, axis=1)
        det_total += (-1) ** coluna * matriz[0, coluna] * determinante_matriz(menor)

    return det_total

# Solicitar ao usuário a dimensão da matriz
dimensao = int(input("Digite a dimensão da matriz: "))

# Solicitar ao usuário os elementos da matriz
print("Digite os elementos da matriz:")
matriz_usuario = np.array([[float(input(f"Digite o elemento [{i + 1}][{j + 1}]: ")) for j in range(dimensao)] for i in range(dimensao)])

# Calcular determinante da matriz original
determinante_original = determinante_matriz(matriz_usuario)

# Calcular determinante do menor da matriz 2x2
menor_2x2 = matriz_usuario[:2, :2]
determinante_menor_2x2 = determinante_matriz(menor_2x2)

# Calcular determinante do menor da matriz 1x1 dentro da matriz original
menor_1x1 = matriz_usuario[0, 0]
determinante_menor_1x1 = determinante_matriz(np.array([[menor_1x1]]))

# Verificar se a matriz é definida positiva ou negativa definida
definicao = ""
if all(determinante > 0 for determinante in [determinante_original, determinante_menor_2x2, determinante_menor_1x1]):
    definicao = "positiva definida"
elif all(determinante % 2 == 0 for determinante in range(len([determinante_original, determinante_menor_2x2, determinante_menor_1x1]))):
    definicao = "negativa definida"
else:
    definicao = "indefinida"

print("\nA matriz é", definicao)
