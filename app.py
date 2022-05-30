from flask import Flask, render_template, request
from test import predict_model

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def prediction():
    if request.method == 'POST':
        Alcohol = request.form["Alcohol"]
        Malic_Acid = request.form["Malic_Acid"]
        result = predict_model(Alcohol, Malic_Acid)
        return render_template("predict.html",result = result )

if __name__ == '__main__':
    app.run(debug=True)