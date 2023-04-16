from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='templates')

# Load the trained model from file
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    count_contraction = float(request.form['count_contraction'])
    length_contraction = float(request.form['length_contraction'])
    std = float(request.form['std'])
    entropy = float(request.form['entropy'])
    contradiction_times = float(request.form['contradiction_times'])

    # Make a prediction using the pre-trained model
    prediction = model.predict([[count_contraction, length_contraction, std, entropy, contradiction_times]])

    # Render the prediction result in a new page
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
