from transformers import AutoModelForQuestionAnswering, pipeline, AutoTokenizer
import numpy as np
import pandas as pd
import os
import warnings
from flask import Flask, render_template, request, flash, Markup
warnings.filterwarnings("ignore")
from chatBot import ChatBot


app = Flask(__name__)
app.secret_key = "mamabear333"


@app.route("/ai")
def index():
    flash("")
    return render_template("index.html")

#initialization of ChatBot class
NAMES = ["models/roberta-base-squad2"]
CONTEXT_PATH = "data/context/day5_context_cleanup.txt"

# create ChatBot object, import names list and read the context
cb = ChatBot(NAMES)
cb.read_file(CONTEXT_PATH)


# @app.route("/answer", methods=["POST"])
# def answer():
#     form_submission = str(request.form['name_input'])
#     question = form_submission
#     response = cb.single_model_answer(question)
#     flash("Question: "+question +"   Answer: "+response)
#     # flash(response)
#     return render_template("index.html")
@app.route("/answer", methods=["POST"])
def answer():
    form_submission = str(request.form["question_input"])
    question = form_submission
    response = cb.single_model_answer(question)

    flash(Markup("Q: " + question + "<br> A: " + response))
    return render_template("index.html")
