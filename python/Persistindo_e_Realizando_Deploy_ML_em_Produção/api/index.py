import numpy as np
import os
from flask import Flask, request, render_template, make_response #cria aplicação web
#from sklearn.externals import joblib

import pickle

app = Flask(__name__, static_url_path='/assets')  #inicializar a aplicação
#model = joblib.load('model/model.pkl')  #carregar o modelo
with open('modelo/modelo2.pkl', 'rb') as f:
    model = pickle.load(f)
    
# rota 1
@app.route('/')
def display_gui():
    return render_template('template.html')

# rota 2
@app.route('/verificar', methods=['POST'])
def verificar():
	Pregnancies              = request.form['Pregnancies']
	Glucose                  = request.form['Glucose']
	BloodPressure            = request.form['BloodPressure']
	SkinThickness            = request.form['SkinThickness']
	Insulin                  = request.form['Insulin']
	BMI                      = request.form['BMI']
	DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
	Age                      = request.form['Age']
    
	teste = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
	
    # Os dados abaixo pode ser salvos em um banco de dados para futuras analises. Pode pegar o IP, força o usuário fazer um login
    # estou mostrando para fins didáticos
	print(":::::: Dados de Teste ::::::")
	print("Pregnancies: {}".format(Pregnancies))
	print("Glucose: {}".format(Glucose))
	print("BloodPressure: {}".format(BloodPressure))
	print("SkinThickness: {}".format(SkinThickness))
	print("Insulin: {}".format(Insulin))
	print("BMI: {}".format(BMI))
	print("DiabetesPedigreeFunction: {}".format(DiabetesPedigreeFunction))
	print("Age: {}".format(Age))
	print("\n")

	classe = model.predict(teste)[0]
	print("Classe Predita: {}".format(str(classe)))

	return render_template('template.html',classe=str(classe))

if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5500))
        app.run(host='0.0.0.0', port=port)

