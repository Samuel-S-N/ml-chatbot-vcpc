import numpy as np
import matplotlib


# 1. DEFINIÇÃO DOS DADOS - o que aconteceu na realidade vs o que o modelo previu.
y_real = np.array([10.0, 20.0, 30.0]) # y_real representa os valores verdadeiros

y_hat = np.array([12.0, 18.0, 40.0]) # y_hat representa o valor PREVISTO pelo modelo


# 2. CÁLCULO DOS RESÍDUOS
# Subtraímos o valor real do previsto para encontrar o erro de cada ponto

residuos = y_real - y_hat


# 3. ALGORITMO DO MAE (Mean Absolute Error)

# Passo A: Transformamos todos os erros em valores positivos (absolutos)
erros_absolutos = np.abs(residuos)

# Passo B: Calculamos a média desses valores positivos
mae = np.mean(erros_absolutos)

# 4. ALGORITMO DO RMSE (Root Mean Squared Error)
# Passo A: Elevamos os erros ao quadrado
erros_quadraticos = residuos ** 2

# Passo B: Calculamos a média dos erros quadráticos (MSE)
mse = np.mean(erros_quadraticos)

# Passo C: Extraímos a raiz quadrada para voltar à unidade de medida original
rmse = np.sqrt(mse)


# 5. EXIBIÇÃO DOS RESULTADOS
print("--- Demonstração do Algoritmo ---")
print(f"Valores Reais (y): {y_real}")
print(f"Valores Previstos (ŷ): {y_hat}")
print("-" * 33)
print(f"MAE calculado: {mae:.2f}")
print(f"RMSE calculado: {rmse:.2f}")