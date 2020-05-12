from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import numpy as np



file = open('model_MLP.pkl', 'rb')

mlp = pickle.load(file)
file.close()
@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = float(myDict['fever'])
        tire = int(myDict['tire'])
        cough = int(myDict['cough'])
        breath = int(myDict['breath'])
        throat = int(myDict['throat'])
        pain = int(myDict['pain'])
        nasal = int(myDict['nasal'])
        rnose = int(myDict['rnose'])
        age = int(myDict['age'])

        a0=0
        a1=0
        a2=0
        a3=0
        a4=0

        if age==1:
            a0=1
        elif age==2:
            a1=1
        elif age==3:
            a2=1       
        elif age==4:
            a3=1
        elif age==5:
            a4=1

       
        # inputFeatures = [fever, pain, age, runnyNose, diffBreath]
        # O_test = np.array([[fever, pain, age, runnyNose, diffBreath]])
        # O_test = np.array([[0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0]])

        O_test = np.array([[fever,tire,cough,breath,throat,0,pain,nasal,rnose,a0,a1,a2,a3,a4,0,0,0]])

        y_pred = mlp.predict(O_test)
        # infProb =mlp.predict_proba([inputFeatures])[0][1]

        print("Test Case")

        print(O_test)
        
        print("Prediction")

        print(y_pred)
        # print(infProb)
        print(mlp.predict_proba(O_test))
        inff =mlp.predict_proba(O_test)[0][1]
     

        return render_template('show.html', inf= round(inff*100))
    return render_template('index.html')
    
   # return 'Hello, World!' + str(infProb)



if __name__=="__main__":
    app.run(debug=True)