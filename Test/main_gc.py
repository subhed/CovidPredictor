
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import numpy as np



file = open('model_GBC.pkl', 'rb')

gbc = pickle.load(file)
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

        O_test_x = np.array([[fever,tire,cough,breath,throat,0,pain,nasal,rnose,a0,a1,a2,a3,a4,0,0,0]])
        O_test_y = np.array([[1]])

        ypred = gbc.predict(O_test_x)
        # infProb =mlp.predict_proba([inputFeatures])[0][1]

        print("Prediction :", ypred)
        acc=gbc.score(O_test_x, O_test_y) 
        print("Accuracy: ", acc)
        print(gbc.predict_proba(O_test_x))
            
        ans = "No"

        if ypred[0]==1:
            ans="Yes"
        elif ypred[0]==0:
            ans="No"

        return render_template('show_gc.html', inf= (ans))
    return render_template('index_gc.html')
    
   # return 'Hello, World!' + str(infProb)



if __name__=="__main__":
    app.run(debug=True)


# find optimal learning rate value


