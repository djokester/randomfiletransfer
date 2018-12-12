import json
import requests
import pandas as pd
from multiprocessing import Process
import sys
import time
from flask_cors import CORS
import flask
from pycorenlp.corenlp import StanfordCoreNLP
from pprint import pprint 

app = flask.Flask(__name__)
CORS(app)


 
host = "http://localhost"
port = "9000"
nlp = StanfordCoreNLP(host + ":" + port)

def classify(text):
    if text in ["JJ", "JJR", "JJS"]:
        return("Descriptor")
    if text in ["NN", "NNP","NNPS", "NNS"]:
        return("Entity")
    if text in ["VB", "VBD", "VBG", "VBZ", "VBN", "VBP"]:
        return("Action/Service")
    
def pos(text):
    host = "http://localhost"
    port = "9000" 
    nlp = StanfordCoreNLP(host + ":" + port)
    lst = []
    print("POS", text)
    output = nlp.annotate(
    text,
    properties={
        "outputFormat": "json",
        "annotators": "pos"
    })
    #output = (output["sentences"][0]["tokens"])
    lst2 = []
    for i in output["sentences"]:
        lst2 = lst2 + i["tokens"]
    #print(output)
    interest = ["JJ", "JJR", "JJS", "NN", "NNP","NNPS", "NNS", "VB", "VBD", "VBG", "VBZ", "VBN", "VBP"]
    
    for i in lst2:
        if i["pos"] in interest:
            lst.append([i["originalText"], classify(i["pos"])])
    print("POS OUTPUT", lst)
    return(lst)
    #return(output)

def ner(text):
    
    host = "http://localhost"
    port = "9000"
    nlp = StanfordCoreNLP(host + ":" + port)
    lst = []
    output = nlp.annotate(
    text,
    properties={
        "outputFormat": "json",
        "annotators": "ner"
    })
    #output = (output["sentences"][0]["tokens"])
    #output = (output["sentences"][0]['entitymentions'])
    lst2 = []
    for i in output["sentences"]:
        lst2 = lst2 + i["entitymentions"]
    for i in lst2:
        lst.append([i["text"], i["ner"]])
    return(lst)
    print("NER", lst)
    #return(output)

def sentiment(text):
    host = "http://localhost"
    port = "9000"
    nlp = StanfordCoreNLP(host + ":" + port)
    lst = []
    output = nlp.annotate(
    text,
    properties={
        "outputFormat": "json",
        "annotators": "sentiment"
    })
    #output = (output["sentences"][0]["tokens"])
    #output = (output["sentences"][0]['entitymentions'])
    """for i in output:
        lst.append([i["text"], i["ner"]])
    return(lst)"""
    return(output)

@app.route("/predict", methods=["POST"])
def predict():
    print("Initialised Predict")
    #print(flask.request.method)
    data = {"success": False}
    if flask.request.method == "POST":
        input_data = flask.request.get_json()
        print(input_data)
        query = input_data["query"] 
        parts_of_speech = pos(query)
        named_entities = ner(query)
        #x["measures"] = out[0]
        print(parts_of_speech)
        data["predictions"] = pos(query) + ner(query)
        data["success"] = True
        print(json.dumps(data))
    return flask.jsonify(data)
    
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    #loadModel()
    app.run(host='0.0.0.0', port = 8000)
    
