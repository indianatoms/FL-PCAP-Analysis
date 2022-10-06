from unittest import result
from flask import request
from flaskapp.models import User, RegresionParameters
from flaskapp import app, db, bcrypt
from flask import json, abort, make_response, jsonify
from marshmallow import Schema, fields, ValidationError


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
        response=json.dumps({"result":"success"}),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/local_model', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        print(json['name']) 
        return json
    else:
        return 'Content-Type not supported!'
