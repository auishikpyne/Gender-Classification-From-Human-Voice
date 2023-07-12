from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils import create_model
import os
from infer_new import gender_pred

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@app.route("/gender_prediction/", methods=["POST"])
def gender_prediction():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    file = request.files["file"]
    
    if file:
        # Perform audio processing or save the file as needed
        file_path = os.path.join("saved_data", secure_filename(file.filename))
        file.save(file_path)
        # file_path = '/home/auishik/gender_classification/gender-recognition-by-voice/saved_data/tusher.wav'
        result = gender_pred(file_path)
        print(result)
        
        return jsonify(result)
    
    return "No file provided."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7777, debug=True)
