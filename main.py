from flask import Flask, redirect, url_for, render_template, request, Response
from Utilities.volume_predictor import volume_predictor

app = Flask(__name__)


# create welcome page for user
@app.route("/")
def welcome():
    return render_template("index.html")


# Response processing after user response submission
@app.route("/submit", methods=["POST", "GET"])
def submit():
    if request.method == "POST":
        Moving_average = request.form["MA"]
        Rolling_median = request.form["AJ"]
        test_values, mov_avg, roll_med = volume_predictor(
            Moving_average, Rolling_median
        )
        return render_template(
            "result.html",
            test_values=test_values,
            mov_avg=Moving_average,
            roll_med=Rolling_median,
        )


if __name__ == "__main__":
    app.run(debug=True)
