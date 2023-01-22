import datetime
from flask import request, jsonify
import uuid
from flaskapp.models import User, Model
from flaskapp import app, db, bcrypt
from flask import json, make_response
from marshmallow import Schema, fields, ValidationError
import numpy as np
import jwt
from functools import wraps


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        if not token:
            return jsonify({'message' : 'Token is missing!'}), 401

        try: 
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message' : 'Token is invalid!'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


@app.route("/ping")
def hello_world():
    return jsonify({"message": "pong"})


@app.route("/login")
def login():
    auth = request.authorization

    if not auth or not auth.username or not auth.password:
        return make_response(
            "Could not verify",
            401,
            {"WWW-Authenticate": 'Basic realm="Login required!"'},
        )

    user = User.query.filter_by(username=auth.username).first()

    if not user:
        return make_response(
            "Could not verify",
            401,
            {"WWW-Authenticate": 'Basic realm="Login required!"'},
        )

    if bcrypt.check_password_hash(user.password, auth.password):
        token = jwt.encode(
            {
                "public_id": user.public_id,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
            },
                app.config['SECRET_KEY']
            )
        return jsonify({'token' : token})

    return make_response(
            "Could not verify",
            401,
            {"WWW-Authenticate": 'Basic realm="Login required!"'},
        )


### Schemas ###
class ParametersSchema(Schema):
    intercept = fields.Float(required=True)
    coefs = fields.List(fields.Float(default=""))
    dataset_size = fields.Integer(required=True)
class ModelSchema(Schema):
    type = fields.String(required=True)
    model = fields.Nested(ParametersSchema)
class RegisterSchema(Schema):
    username = fields.String(required=True)
    email = fields.String(required=True)
    password = fields.String(required=True)


###TOKEN REQUIRED ZONE ###
@app.route("/user", methods=["GET"])
@token_required
def get_all_users(current_user):

    if not current_user.admin:
        return jsonify({'message':'Cannot perform that. Only and admin function.'})

    users = User.query.all()
    output = []
    for user in users:
        user_data = {}
        user_data["public_id"] = user.public_id
        user_data["name"] = user.username
        user_data["admin"] = user.admin
        output.append(user_data)
    return jsonify({"users": output})


@app.route("/user/<public_id>", methods=["GET"])
@token_required
def get_one_users(current_user ,public_id):

    if not current_user.admin:
        return jsonify({'message':'Cannot perform that. Only and admin function.'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({"message": "No user found!"})

    user_data = {}
    user_data["public_id"] = user.public_id
    user_data["name"] = user.username
    user_data["admin"] = user.admin

    return jsonify({"user": user_data})


@app.route("/user", methods=["POST"])
# @token_required
def register():

    # if not current_user.admin:
    #     return jsonify({'message':'Cannot perform that. Only and admin function.'})

    data = request.json
    schema = RegisterSchema()

    try:
        # Validate request body against schema data types
        schema.load(data)
    except ValidationError as err:
        # Return a nice message if validation fails
        return jsonify(err.messages), 400

    hashed_password = bcrypt.generate_password_hash(data["password"])

    user = User(
        public_id=str(uuid.uuid4()),
        username=data["username"],
        email=data["email"],
        password=hashed_password,
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "New user created."})


@app.route("/user/<public_id>", methods=["PUT"])
@token_required
def promote_user(current_user, public_id):

    if not current_user.admin:
        return jsonify({'message':'Cannot perform that. Only and admin function.'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({"message": "No user found!"})

    user.admin = True
    db.session.commit()

    return jsonify({"message": "The user has been promoted."})


@app.route("/user/<public_id>", methods=["DELETE"])
@token_required
def delete_user(current_user, public_id):

    if not current_user.admin:
        return jsonify({'message':'Cannot perform that. Only and admin function.'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({"message": "No user found!"})

    db.session.delete(user)
    db.session.commit()

    return jsonify({"message": "The user has been deleted."})




### MODELS ###

@app.route("/model", methods=["GET"])
@token_required
def get_all_models(current_user):
    if not current_user.admin:
        return jsonify({'message':'Cannot perform that. Only an admin function.'})
    reg_models = Model.query.all()

    output = []
    for model in reg_models:
        model_data = {}
        model_data['id'] = model.id
        model_data["model_type"] = model.model_type
        model_data["global_model"] = model.global_model
        model_data["model"] = model.model_parameters
        output.append(model_data)
    return jsonify({"models": output})


@app.route("/model", methods=["POST"])
@token_required
def add_model(current_user):

    data = request.json
    # schema = ModelSchema()

    # try:
    #     # Validate request body against schema data types
    #     schema.load(data)
    # except ValidationError as err:
    #     # Return a nice message if validation fails
    #     return jsonify(err.messages), 400

    model = Model(
        model_type=data["type"],
        model_parameters=data["model"],
        user_id=current_user.id,
        global_model=False,
    )
    db.session.add(model)
    db.session.commit()

    return jsonify({"message":"Model created"})


@app.route("/model/<int:model_id>", methods=['GET'])
@token_required
def post(current_user, model_id):
    model = Model.query.get_or_404(model_id)

    if model.user_id == current_user.id or current_user.admin:
        response = app.response_class(
            response=json.dumps(
                {
                    "model_id": model.id,
                    "model_type": model.model_type,
                    "new": model.new,
                }
            ),
            status=200,
            mimetype="application/json",
        )
    return jsonify({"message":"You do not have access to thsi model"}), 403


@app.route("/model/<int:model_id>", methods=["DELETE"])
@token_required
def delete_post(current_user, model_id):
    model = Model.query.get_or_404(model_id)

    if model.user_id == current_user.id or current_user.admin:
        db.session.delete(model)
        db.session.commit()
        response = app.response_class(
            response=json.dumps({"result": "successfully deleted model"}),
            status=200,
            mimetype="application/json",
        )
        return response
    return jsonify({"message":"You do not have access to thsi model"}), 403

@app.route("/user/<user_public_id>/models", methods=["GET"])
@token_required
def display_models(current_user, user_public_id):
    user = User.query.get_or_404(user_public_id)

    if user.public_id == current_user.public_id or current_user.admin:

        output = []
        for model in user.regModel:
            model_data = {}
            model_data["intercept"] = model.public_id
            model_data["bias"] = model.username
            model_data["new"] = model.admin
            model_data["global_model"] = model.admin
            model_data["date"] = model.date_posted
            output.append(model_data)

        return jsonify({"user": user_public_id, "models": output})
    return jsonify({"message":"You do not have access to thsi model"}), 403


@app.route("/model/global/calculate", methods=["GET"])  # only allow for admin
@token_required
def calcualte_global_model(current_user):

    if not current_user.admin:
        return jsonify({'message':'Cannot perform that. Only an admin function.'})
    
    data = request.json
    sum_intercept = 0
    model = Model.query.get_or_404(data["models_id"][0])
    model_type = model.model_type
    sum_coefs = np.zeros(len(model.model_parameters['coefs']))
    for id in data["models_id"]:
        model = Model.query.get_or_404(id)
        sum_intercept = sum_intercept + model.model_parameters['intercept']
        sum_coefs = np.add(sum_coefs, model.model_parameters['coefs'])
        db.session.commit()

    avg_intercept = sum_intercept / len(data["models_id"])
    avg_coefs = sum_coefs / len(data["models_id"])

    model_json = { "intercept": avg_intercept, "coefs":avg_coefs.tolist() }

    model = Model(
        model_parameters=model_json, model_type=model_type ,global_model=True, user_id=current_user.id
    )
    db.session.add(model)
    db.session.commit()

    response = app.response_class(
        response=json.dumps(
            {"intercept": avg_intercept, "bias": avg_coefs.tolist(), "id": model.id}
        ),
        status=200,
        mimetype="application/json",
    )
    return response

@app.route("/model/global/calculate/fedavg", methods=["GET"])  # only allow for admin
@token_required
def calcualte_fedavg_model(current_user):

    if not current_user.admin:
        return jsonify({'message':'Cannot perform that. Only and admin function.'})
    
    data = request.json
    
    model = Model.query.get_or_404(data["models_id"][0])
    model_type = model.model_type

    if model_type == "SGDClassifier" or model_type == "LogisticRegression":
        sum_intercept = 0
        sum_coefs = np.zeros(len(model.model_parameters['coefs']))
        dataset_total = 0
    if model_type == "SGDClassifier" or model_type == "LogisticRegression":
        for id in data["models_id"]:
            model = Model.query.get_or_404(id)
            sum_intercept += model.model_parameters['intercept'] * model.model_parameters['dataset_size']
            sum_coefs = np.add(sum_coefs, np.array(model.model_parameters['coefs']) * model.model_parameters['dataset_size']) 
            dataset_total += model.model_parameters['dataset_size']

        avg_intercept = sum_intercept / dataset_total
        avg_coefs = sum_coefs / dataset_total

        model_json = { "intercept": avg_intercept, "coefs":avg_coefs.tolist() }

    if model_type == "NeuralNetworkClassifier":
        fc1_bias = np.zeros(np.array(model.model_parameters['intercept'][0]).shape)
        fc2_bias = np.zeros(np.array(model.model_parameters['intercept'][1]).shape)
        # fc3_bias = np.zeros(np.array(model.model_parameters['intercept'][2]).shape)

        fc1_weights = np.zeros(np.array(model.model_parameters['coefs'][0]).shape)
        fc2_weights = np.zeros(np.array(model.model_parameters['coefs'][1]).shape)
        # fc3_weights = np.zeros(np.array(model.model_parameters['coefs'][2]).shape)

        dataset_total = 0

    if model_type == "NeuralNetworkClassifier":
        for id in data["models_id"]:
            model = Model.query.get_or_404(id)

            fc1_bias += np.array(model.model_parameters['intercept'][0]) * model.model_parameters['dataset_size'] 
            fc2_bias += np.array(model.model_parameters['intercept'][1]) * model.model_parameters['dataset_size']
            # fc3_bias += np.array(model.model_parameters['intercept'][2]) * model.model_parameters['dataset_size']

            fc1_weights += np.array(model.model_parameters['coefs'][0]) * model.model_parameters['dataset_size']
            fc2_weights += np.array(model.model_parameters['coefs'][1]) * model.model_parameters['dataset_size']
            # fc3_weights += np.array(model.model_parameters['coefs'][2]) * model.model_parameters['dataset_size']

            dataset_total += model.model_parameters['dataset_size']
        
        fc1_bias_avg = fc1_bias / dataset_total 
        fc2_bias_avg = fc2_bias / dataset_total
        # fc3_bias_avg = fc3_bias / dataset_total

        fc1_weight_avg = fc1_weights / dataset_total
        fc2_weight_avg = fc2_weights / dataset_total
        # fc3_weight_avg = fc3_weights / dataset_total

        model_json = { "intercept": [fc1_bias_avg.tolist(),fc2_bias_avg.tolist()], "coefs": [fc1_weight_avg.tolist(), fc2_weight_avg.tolist()] }


    model = Model(
        model_parameters=model_json, model_type=model_type ,global_model=True, user_id=current_user.id
    )
    db.session.add(model)
    db.session.commit()

    response = app.response_class(
        response=json.dumps(
            {"stauts": "model created succesfully", "id": model.id}
        ),
        status=200,
        mimetype="application/json",
    )
    return response


@app.route("/model/global", methods=["GET"])  # for all
@token_required
def get_global_model(current_user):
    model = Model.query.filter_by(global_model=True).first()
    
    response = app.response_class(
        response=json.dumps({"models": model.model_parameters}),
        status=200,
        mimetype="application/json",
    )
    return response
