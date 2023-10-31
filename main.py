import pickle
from flask import Flask , request ,jsonify,render_template
import numpy as np
import pandas as pd


application = Flask(__name__)
app = application

data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))


# Route for home page
@app.route('/')
def index():
    
    locations = sorted(data['location'].unique())
    return render_template('home.html', locations= locations)

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        location = request.form.get('location')
        total_sqft = request.form.get('total_sqft')
        bath = request.form.get('bath')
        bhk = request.form.get('bhk')

        
        input = pd.DataFrame(location, total_sqft, bath, bhk, columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input)[0] *1e5
        result = str(np.round(prediction,2))
        print(result)
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host = "0.0.0.0")