from crypt import methods
from unittest import result
from flask import request
from flaskapp.models import User, RegresionParameters
from flaskapp import app, db, bcrypt
from flask import json, abort, make_response, jsonify
from marshmallow import Schema, fields, ValidationError
import numpy as np



@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


class RegisterSchema(Schema):
    username = fields.String(required=True)
    email = fields.String(required=True)
    password = fields.String(required=True)

def validate_username(username):
    user = User.query.filter_by(username=username).first()
    if user:
        msg = jsonify(message="User")
        response = make_response(msg, 400)
        abort(response)


@app.route("/register", methods=['POST'])
def register():

    data = request.json
    schema = RegisterSchema()
    
    try:
        # Validate request body against schema data types
        schema.load(data)
    except ValidationError as err:
        # Return a nice message if validation fails
        return jsonify(err.messages), 400

    validate_username(data['username'])
    hashed_password = bcrypt.generate_password_hash(data['password'])

    user = User(username=data['username'], email=data['email'], password = hashed_password)
    db.session.add(user)
    db.session.commit()
    response = app.response_class(
        response=json.dumps({"id":user.id}),
        status=200,
        mimetype='application/json'
    )
    return response

class RegresionModelSchema(Schema):
    intercept = fields.Float(required=True)
    bias = fields.List(fields.Float(default=''))
    author = fields.Integer(required=True)


@app.route('/local_model', methods=['POST'])
def process_json():
    
    data = request.json
    schema = RegresionModelSchema()
    
    try:
        # Validate request body against schema data types
        schema.load(data)
    except ValidationError as err:
        # Return a nice message if validation fails
        return jsonify(err.messages), 400

    model = RegresionParameters(intercept = data['intercept'],bias = data['bias'],
    user_id=data['author'], global_model=False)
    db.session.add(model)
    db.session.commit()
    
    response = app.response_class(
        response=json.dumps({"result":"model added"}),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/model/<int:model_id>")
def post(model_id):
    model = RegresionParameters.query.get_or_404(model_id)
    response = app.response_class(
        response=json.dumps({"model_id" : model.id,
        "bias" : model.bias,
        "intercept" : model.intercept,
        "new":model.new
                                                   }),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/model/<int:model_id>/delete", methods=['DELETE'])
def delete_post(model_id):
    model = RegresionParameters.query.get_or_404(model_id)
    db.session.delete(model)
    db.session.commit()
    response = app.response_class(
        response=json.dumps({"result":"successfully deleted model"}),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/user/<int:user_id>/models", methods=['GET'])
def display_models(user_id):
    user = User.query.get_or_404(user_id)
    result = [(d.id, d.new) for d in user.regModel]
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/model/global/calculate", methods=["GET"]) #only allow for admin
def calcualte_global_model():
    data = request.json
    sum_intercept = 0
    model = RegresionParameters.query.get_or_404(data['models_id'][0])
    sum_bias = np.zeros(len(model.bias))
    for id in data['models_id']:
        model = RegresionParameters.query.get_or_404(id)
        if (model.new == False):
            msg = jsonify(message="Models have already been averaged.")
            response = make_response(msg, 400)
            abort(response)
        sum_intercept = sum_intercept + model.intercept
        sum_bias =+ np.add(sum_bias, model.bias)
        model.new = False
        db.session.commit()

    avg_intercept = sum_intercept/len(data['models_id']) 
    avg_bias = sum_bias/len(data['models_id']) 

    model = RegresionParameters(intercept = avg_intercept,bias = avg_bias.tolist(), global_model=True, user_id = 999)
    db.session.add(model)
    db.session.commit()
    
    response = app.response_class(
        response=json.dumps(
            {"intercept":avg_intercept,
             "bias": avg_bias.tolist(),
             "id":model.id}
             ),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/model/global", methods=["GET"]) #for all
def get_global_model():
    model = RegresionParameters.query.filter_by(global_model=True,new=True).first()
    response = app.response_class(
        response=json.dumps(
            {"intercept":model.intercept,
             "bias": model.bias}),
        status=200,
        mimetype='application/json'
    )
    return response

