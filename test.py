from flask import Flask, request, jsonify
from gtts import gTTS

app = Flask(__name__)

# Store text in a global variable
text_storage = {}

@app.route('/send-text', methods=['POST'])
def send_text():
    data = request.json
    text = data.get('text', '')
    text_storage['text'] = text
    return jsonify(message="You have sent a text")

@app.route('/generate-audio', methods=['GET'])
def generate_audio():
    text = text_storage.get('text', '')
    if text:
        tts = gTTS(text=text, lang='en', tld="co.uk")
        audio_path = 'output.mp3'
        tts.save(audio_path)
        return jsonify(message="Audio file generated", audio_file=audio_path)
    else:
        return jsonify(message="No text available"), 400

if __name__ == '__main__':
    app.run(debug=True)
