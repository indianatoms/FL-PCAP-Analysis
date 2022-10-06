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
    global_model = db.Column(db.Boolean, unique=False, default=False)
    intercept = db.Column(db.Float ,nullable=False)
    bias = db.Column(db.PickleType)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Reg Model('{self.id}','{self.intercept}','{self.new}'),'{self.global_model}')"
