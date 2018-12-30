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
    if text["pos"] in ["JJ", "JJR", "JJS"]:
        return("Descriptor")
    else:
        if text["pos"] in ["NN", "NNP","NNPS", "NNS", "CD"] or text["originalText"].lower() == "am":
            return("Entity")
        else:
            if text["pos"] in ["VB", "VBD", "VBG", "VBZ", "VBN", "VBP"] or text["originalText"] == "n't" or text["originalText"] == "not":
                return("Action/Service")
    
def pos(text):
    host = "http://localhost"
    port = "9000" 
    nlp = StanfordCoreNLP(host + ":" + port)
    lst = []
    #print("POS", text)
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
    interest = ["JJ", "JJR", "JJS", "NN", "NNP","NNPS", "NNS", "VB", "VBD", "VBG", "VBZ", "VBN", "VBP", "CD"]
    
    for i in lst2:
        if i["pos"] in interest or i["originalText"] == "n't" or i["originalText"] == "not":
            lst.append([i["originalText"], i["index"], classify(i)])
    #print("POS OUTPUT", lst)
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
        #print(input_data)
        query = input_data["query"]
        print(query)
        parts_of_speech = pos(query)
        named_entities = ner(query)
        #x["measures"] = out[0]
        #print(named_entities)
        predictions = pos(query) + ner(query)
        #print(predictions)
        output = []
        counter = 0
        while counter <= len(parts_of_speech)-1:
            item = parts_of_speech[counter]
            index = counter
            temp = [item]
            count = item[1]
            #print(temp)
            if counter == len(parts_of_speech)-1:
                output.append(temp)
                break
            else:
                for i in parts_of_speech[index+1:]:
                    if i[1] == count+1:
                        temp.append(i)
                        count+=1
                        counter = parts_of_speech.index(i)
                        if counter == len(parts_of_speech)-1:
                            counter = counter+1
                    else:
                        counter = counter+1
                        break
                
            output.append(temp)
        #print(output)
        output2 = []
        for i in output:
            temp = [j[0] for j in i]
            temp = " ".join(temp)
            temp2 = [j[2] for j in i]
            if len(list(set(temp2))) >= 2:
                if len(list(set(temp2))) == 2:
                    if "Entity" in (list(set(temp2))) and "Action/Service" in (list(set(temp2))):
                        descriptor = "Action/Service"
                    if "Entity" in (list(set(temp2))) and "Descriptor" in (list(set(temp2))):
                        descriptor = "Entity"
                    if "Action/Service" in (list(set(temp2))) and "Descriptor" in (list(set(temp2))):
                        descriptor = "Action/Service"
                else:
                    descriptor = "Entity"
            else:
                descriptor = list(set(temp2))[0]
            output2.append([temp, descriptor])
        output2 = [i for i in output2 if i[0].lower() != "nixie"]
        #print(output2)
        entities = [i[0] for i in output2 if i[1]=="Entity"]
        services = [i[0] for i in output2 if i[1]=="Action/Service"]
        #print(entities)
        #print("The entities are " + ",".join(entities))
        f = open("index.html","w")
        f.write("<html>\n" + "<p> <b>Entities : </b>" +  "  ||  ".join(entities) + "</p>" + "\n<p><b>Action : </b>"  + "  ||  ".join(services)+ "</p>\n</html>")
        f.close()
        
        data = {"success": True}
        data["entities"] = "The entities are " + ",".join(entities)
        data["services"] = "The services are " + ",".join(services)
        print(json.dumps(data))
    return flask.jsonify(data)
    
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    #loadModel()
    app.run(host='0.0.0.0', port = 8080)
    
