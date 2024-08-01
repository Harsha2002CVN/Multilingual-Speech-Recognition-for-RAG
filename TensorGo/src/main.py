from transcribe import transcribe_audio
from translate import translate_text
from query_rag import query_rag
from dummy_document import initialize_dummy_document

def main(audio_path):
    # Initialize the dummy document
    initialize_dummy_document()

    # Step 1: Transcribe the audio
    transcription = transcribe_audio(audio_path)
    print("Transcription:", transcription)

    # Step 2: Translate the transcription if necessary
    translated_text = translate_text(transcription)
    print("Translated Text:", translated_text)

    # Step 3: Query the RAG model with the translated text
    rag_output = query_rag(translated_text)
    print("RAG Output:", rag_output)

if __name__ == "__main__":
    audio_path = "Sam.mp3"
    main(audio_path)
