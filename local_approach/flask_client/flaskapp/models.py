from datetime import datetime
from email.policy import default
from enum import unique
from flaskapp import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    address = db.Column(db.String(20), unique=True, nullable=False)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    admin = db.Column(db.Boolean, default=False)

    regModel = db.relationship('Model', backref='author', lazy=True) #class reference

    def __repr__(self):
        return f"User('{self.username}','{self.email}')"

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), unique=True, nullable=False)
    global_model = db.Column(db.Boolean, unique=False, default=False)
    model = db.Column(db.PickleType)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Reg Model('{self.id}','{self.intercept}','{self.new}'),'{self.global_model}')"
