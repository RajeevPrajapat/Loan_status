from flask import Flask, render_template, request, Response
import pandas as pd
import numpy as np
import pickle 

application = Flask(__name__)
app = application
# Load Scaler and Model
Scaler = pickle.load(open('model/standerScaler.pkl', 'rb'))
model = pickle.load(open('model/modelForPrediction.pkl', 'rb')) 

# Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route for Single Data Point Prediction
@app.route('/check', methods=['POST','GET'])
def predict_data():
    if request.method == 'POST':
        try:
            # Get form data
            data = [float(x) for x in request.form.values()]
            print(data)
            final_input = Scaler.transform(np.array(data).reshape(1, -1))
            print(final_input)
            prediction = model.predict(final_input)
            print(prediction)
            
            # Determine result class and text
            result_class = 'approved' if prediction[0] == 0 else 'deined'
            result_text = 'Approved' if prediction[0] == 0 else 'Deined'
            
            return render_template('single_prediction.html', result_class=result_class, result_text=result_text)
        except Exception as e:
            return Response(f"Error occurred: {str(e)}", status=500)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)