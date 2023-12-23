from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    if request.method == "POST":
        text = request.form["text"]
        vec = cv.transform([text]).toarray()
        result = model.predict(vec)
        if result[0] == 0:
            prediction_text = "Ham"
        else:
            prediction_text = "Spam"
        return render_template("prediction.html",
                               prediction_text=prediction_text,
                               text=text)


if __name__ == "__main__":
    app.run()