# coding: utf-8
# [START app]

# Default modules
import json
import logging
import os

# Additional modules
import flask
from google.cloud import storage
from google.cloud.storage import Blob
import pickle

# My modules
import ngram
import classifier


# Config
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# Interact with blob in storage (project:sandbox-akihisatakeda)
client = storage.Client()
bucket = client.get_bucket('ana-ocr')

# Load vectorizer
blob_vocab = Blob('vocab.json', bucket)
vocab = json.loads(blob_vocab.download_as_string())
mcv = ngram.MyCountVectorizer()
mcv.load(vocab)

# Load learned model
blob_lrned_model = Blob('lrned_model.pickle', bucket)
pkled_data = blob_lrned_model.download_as_string()
learned_model = pickle.loads(pkled_data) 
mc = classifier.MyClassifier()
mc.load(learned_model)


@app.route('/', methods=['POST']) 
def main():
    req = flask.request.get_json(force=True)
    raw_documents = [i['text'] for i in req['instances']]
    mat = mcv.transform(raw_documents)
    attributes = mc.classify(mat)
    result ={
        'Result' + str(row+1): attributes[row] for row in range(len(attributes))
    }
    return flask.jsonify(result) 


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]