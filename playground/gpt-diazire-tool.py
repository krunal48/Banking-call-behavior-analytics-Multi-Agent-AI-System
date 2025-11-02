import base64
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def transcribe_audio(filepath: str) -> dict:
  client = OpenAI(api_key=OPENAI_API_KEY)

  with open(filepath, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
      model="gpt-4o-transcribe-diarize",
      file=audio_file,
      response_format="diarized_json",
      chunking_strategy="auto",
    )

  modified_output = [
    [segment.start, segment.end, segment.speaker, segment.text.strip()]
    for segment in transcript.segments
]

  print(modified_output) # TODO: Remove in Future, debug print
  print()
  print()
  print()
  print(transcript)
  return {'Transcription': transcript}

transcribe_audio('/home/dedsec995/Downloads/debit_card.wav')

