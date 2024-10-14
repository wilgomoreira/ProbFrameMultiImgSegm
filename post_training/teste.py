import math
import matplotlib.pyplot as plt

# Função do Kernel Gaussiano
def gaussian_kernel(u):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * u**2)

# Função para calcular o KDE
def kde(x, data, h):
    n = len(data)
    result = 0
    for xi in data:
        result += gaussian_kernel((x - xi) / h)
    return result / (n * h)

# Dados de exemplo
data = [1, 2, 3, 4, 5]

# Largura de banda (h)
h = 1

# Gera uma lista de pontos ao longo do eixo x onde queremos calcular a KDE
x_values = [i * 0.1 for i in range(0, 51)]  # de 0 a 5 em incrementos de 0.1

# Calcula o KDE para cada ponto em x_values
kde_values = [kde(x, data, h) for x in x_values]

# Plota o gráfico
plt.plot(x_values, kde_values, label='KDE')
plt.scatter(data, [0] * len(data), color='red', zorder=5, label='Data Points')  # Para mostrar os pontos de dados
plt.title('Kernel Density Estimation')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig(f"post_training/test.png")

new_value = 2.5

# Calcular a densidade do novo valor
new_density = kde(new_value, data, h)

# Exibir o resultado
print(f"Densidade estimada para o valor {new_value}: {new_density}")

