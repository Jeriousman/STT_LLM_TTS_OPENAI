# import GPTTFG.whisper_main as whisper_main
# import pyaudio
import wave 
from threading import Thread, Event
from queue import Queue
import time
import argparse
import numpy as np
####
from openai import OpenAI
import os
from dotenv import load_dotenv

####
from pathlib import Path

load_dotenv(".env")
openai_api_key = os.getenv('OPENAI_API_KEY')


def main(args):

  # input_sound_file_path = "/dev_storage/tory/data/TFG/audios/speech_eng_f_5.wav"
  # input_sound_file_path = "/dev_storage/tory/data/TFG/audios/hama_eng_m_8.wav"
  # output_sound_file_path = Path(__file__).parent / "response_hamma.wav"
  input_sound_file_path = args.input_path
  output_sound_file_path = args.output_path


  
  
  ##STT part
  client = OpenAI(api_key=openai_api_key)
  start_stt = time.time()
  audio_file= open(input_sound_file_path, "rb")
  transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
  )
  time_spent_stt = round(float(time.time() - start_stt), 3)
  print(f"OpenAI STT (whisper-1) took {time_spent_stt} seconds")



  ##now LLM part
  start_llm = time.time()
  model = "gpt-4o"
  completion = client.chat.completions.create(
      model=model,
      messages=[
          {"role":"system", "content": "You are an amazingly supportive assistant. Please answer my questions"},
          {"role":"user", "content": transcription.text},
      ]
  )
  answer =  completion.choices[0].message.content
  time_spent_llm = round(float(time.time() - start_llm), 3)
  print(f"OpenAI-4o took {time_spent_llm} seconds")

  ## TTS part
  start_tts = time.time()
  response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=answer
  )
  response.stream_to_file(output_sound_file_path)
  time_spent_tts = round(float(time.time() - start_tts), 3)
  print(f"OpenAI TTS (tts-1) took {time_spent_tts} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/dev_storage/tory/data/TFG/audios/hama_eng_m_8.wav', type=str, help='path to ffhq dataset in string format')
    parser.add_argument('--output_path', default='/home/hojun/projects/llm/GPTTFG/response_hamma_test.wav', type=str)
    args = parser.parse_args()

    main(args)
    