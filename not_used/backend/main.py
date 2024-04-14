
from flask import request, jsonify
from config import app, db
from models import Contact
from model_1 import main

@app.route('/contacts', methods=['POST'])
def do_the_thing():
    SEX = request.json.get('sex')
    PNEUMONIA = request.json.get('pneumonia')
    AGE = request.json.get('age')
    PREGNANT = request.json.get('pregnant')
    DIABETES = request.json.get('diabetes')
    COPD = request.json.get('copd')
    ASTHMA = request.json.get('asthma')
    INMSUPR = request.json.get('inmsupr')
    HIPERTENSION = request.json.get('hipertension')
    OTHER_DISEASE = request.json.get('otherDisease')
    CARDIOVASCULAR = request.json.get('cardiovascular')
    OBESITY = request.json.get('obesity')
    RENAL_CHRONIC = request.json.get('renalChronic')
    TOBACCO = request.json.get('tobacco')

    # Make score here
    try:
        score = main()
    except Exception as e:
        return jsonify({'message': str(e)}), 400

    return jsonify({'score': score})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)

# localhost:5000/