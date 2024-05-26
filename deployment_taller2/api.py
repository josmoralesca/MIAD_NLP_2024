#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import os
from flask_cors import CORS

os.chdir('D:/MIAD/ML-PNL/GIT/MIAD_NLP_2024/deployment_taller2')
from pred_genres_text_RL import predict_genero

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Clasificacion Generos Peliculas API',
    description='Clasificacion Generos Peliculas API')

ns = api.namespace('clasificacion', 
     description='Clasificacion peliculas')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'TEXTO', 
    type=str, 
    required=True, 
    help='TEXT to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})



@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_genero(args['TEXTO'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
