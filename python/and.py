# Realiza a previsão com os pesos, viés e dados de entrada
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0 # função de ativacão de Heaviside

if __name__ == '__main__':    

	print("Previsão função AND\n")    
	# Classificação do "AND"
	# dados de entrada [X1, X2, Y]
	dataset = [[1,1,1],[1,0,0],[0,1,0],[0,0,0]]

   # viés peso1, peso2
	weights = [-0.5, 0.25, 0.25]

	for row in dataset:
		prediction = predict(row, weights)
		print("Esperado=%d, Previsto=%d" % (row[-1], prediction))

		
		