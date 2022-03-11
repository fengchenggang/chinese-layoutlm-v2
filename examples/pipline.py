# -*- coding: utf-8 -*-
import os
import json
import traceback

import numpy as np
from flask import Flask
from flask import request
from flask import jsonify


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


def kv_extract(bin_file):


@app.route("/kv_extract", methods=['POST','GET'])
def extract():
    try:
        params = request.get_json()
        bin_file = params['file']
        img = kv_extract(bin_file)
        return {'state':'succeed', 'img':img}
    except:
        traceback.print_exc()
        return {"state": 'failed'}

if __name__ == "__main__":
    app.run(host='localhost', port=10005, debug=True)