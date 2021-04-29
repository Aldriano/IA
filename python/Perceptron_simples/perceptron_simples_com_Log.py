import numpy as np
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("log.txt", "w")
class NeuralNetwork():
    
    def __init__(self):
        # Propagação para geração de número aleatório 
        np.random.seed(1)
        
        # Cria matriz de pesos dos neurônios de  3x1 (3 linhas e uma coluna) com  com valores de -1 a 1 e média de 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1 # números aleatórios para garantir sua distribuição eficiente.

    def sigmoid(self, x):
        # Aplicando a função sigmóide que desenha uma curva característica em forma de “S”, como uma função de ativação da rede neural.
        # Esta função pode mapear qualquer valor para um valor de 0 a 1. Ela nos ajudará a normalizar a soma ponderada das entradas
        # Binarizar entre 0 e 1 . Ex. classificar um pessoa boa pagadora ou mal pagadora
        # Forma de converter numeros negativos em 0 e positivo em 1
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Calcular a derivada da  função sigmóide, afim de saber a quantidade apropriada para ajustar (aumentar/reduzir) os pesos (retropropagação).
        # Ajustar os pesos/atualizar de volta
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        # treinar o modelo para fazer previsões precisas enquanto ajusta os pesos continuamente
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)
            
            print(iteration," Entrada: ********************************************************")
            print(training_inputs, "\n Saida calculada = Sigmoid(entrada * peso)\n ",output,"\n")

            # Calculando a taxa de erro para back-propagation
            error = training_outputs - output
            print("Taxa de erro = (Saida calculada - Saida esperada): \nSaida esperada\n",training_outputs,"\nErro:\n",error,"\n")
            
            # Realizando ajustes de peso
            print("\nDerivada da Sigmoid da soma ponderada (entrada * peso): \n",self.sigmoid_derivative(output),"\n")
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            print("\n matriz de erro x a derivada da sigmoid\n", error * self.sigmoid_derivative(output))
            print("\nAjuste dos pesos = transposta da matriz de entrada x (erro x derivada da Sigmoid):\n",adjustments,"\n")

            self.synaptic_weights += adjustments
            print("\nPesos atualizados: \n",adjustments, "\n")

    def think(self, inputs): #função pensar()
        # Passando as entradas através do neurônio para obter a saída
        # Convertendo valores em flutuantes
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
        