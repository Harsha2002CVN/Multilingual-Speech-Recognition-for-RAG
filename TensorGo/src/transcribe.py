import whisper

# Load the pre-trained multilingual Whisper model
model = whisper.load_model("large")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']
