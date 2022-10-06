from datetime import datetime
from flaskapp import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    regModel = db.relationship('RegresionParameters', backref='author', lazy=True) #class reference

    def __repr__(self):
        return f"User('{self.username}','{self.email}')"

class RegresionParameters(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    new = db.Column(db.Boolean, unique=False, default=True)
    intercept = db.Column(db.Integer ,nullable=False)
    bias = db.Column(db.PickleType)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) #table name

    def __repr__(self):
        return f"Reg Model('{self.id}','{self.intercept}')"
