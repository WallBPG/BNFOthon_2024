from config import db
from flask_sqlalchemy import SQLAlchemy

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    SEX = db.Column(db.Integer)
    PNEUMONIA = db.Column(db.Integer)
    AGE = db.Column(db.Integer)
    PREGNANT = db.Column(db.Integer)
    DIABETES = db.Column(db.Integer)
    COPD = db.Column(db.Integer)
    ASTHMA = db.Column(db.Integer)
    INMSUPR = db.Column(db.Integer)
    HIPERTENSION = db.Column(db.Integer)
    OTHER_DISEASE = db.Column(db.Integer)
    CARDIOVASCULAR = db.Column(db.Integer)
    OBESITY = db.Column(db.Integer)
    RENAL_CHRONIC = db.Column(db.Integer)
    TOBACCO = db.Column(db.Integer)

    def to_json(self):
        return {
            'id': self.id
            # More here in camelCase
        }