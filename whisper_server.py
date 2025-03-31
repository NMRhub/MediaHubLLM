import os

import whisper
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Increase timeout to effectively infinite
app.config['TIMEOUT'] = 0

def get_transcription(file_path):
    model = whisper.load_model("/llm_models/whisper/models/large-v3.pt")
    result = model.transcribe(file_path)
    return result["text"]

@app.route('/whisper', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join('/tmp', filename)
    file.save(temp_path)

    try:
        # Call the stub function with the file path
        result = get_transcription(temp_path)

        # Return the result as JSON
        return jsonify(result)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Run the server with a very long timeout
    app.run(host='0.0.0.0', port=11435, threaded=True)
