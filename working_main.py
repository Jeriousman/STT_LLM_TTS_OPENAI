# import GPTTFG.whisper_main as whisper_main
import pyaudio
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

def main(args):

  # input_sound_file_path = "/dev_storage/tory/data/TFG/audios/speech_eng_f_5.wav"
  # input_sound_file_path = "/dev_storage/tory/data/TFG/audios/hama_eng_m_8.wav"
  # output_sound_file_path = Path(__file__).parent / "response_hamma.wav"
  input_sound_file_path = args.input_path
  output_sound_file_path = args.output_path



  ##https://platform.openai.com/docs/guides/speech-to-text/quickstart
  load_dotenv(".env")
  openai_api_key = str = os.getenv('OPENAI_API_KEY')
  client = OpenAI(api_key=openai_api_key)
  start_stt = time.time()
  audio_file= open(input_sound_file_path, "rb")
  transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
  )
  time_spent_stt = round(float(time.time() - start_stt), 3)
  print(f"OpenAI STT (whisper-1) took {time_spent_stt} seconds")
  ### real time microphone recording

  # recording_signals = Queue() ##계속 리코딩할지 스탑할지 시그날을 줌
  # recordings = Queue()


  # p = pyaudio.PyAudio()

  # p.terminate()

  # CHANNELS = 1 ##mono=1, stereo=2
  # FRAME_RATE = 16000
  # FRAMES_PER_BUFFER = 1024
  # RECORD_SECONDS = 20
  # ADUIO_FORMAT = pyaudio.paInt16
  # SAMPLE_SIZE = 2

  # def record_microphone(chunk=FRAMES_PER_BUFFER):
  #     p = pyaudio.PyAudio()
  #     stream = p.open(format=ADUIO_FORMAT,
  #                     channels=CHANNELS,
  #                     rate=FRAME_RATE,
  #                     input=True,
  #                     # input_device_index=,
  #                     frames_per_buffer=FRAMES_PER_BUFFER)
  #     frames = []
  #     while not recording_signals.empty():
  #         data = stream.read(FRAMES_PER_BUFFER)
  #         frames.append(data)

  #         if len(frames) >= (FRAME_RATE*RECORD_SECONDS)/FRAMES_PER_BUFFER: ##소리 받는거 20초동안 다 했으면,
  #             recordings.put(frames.copy())  ##이제 가져온 프레임을 리코딩 queue에 넣은 후
  #             frames = [] 

  #     stream.stop_stream()
  #     stream.close()
  #     p.terminate()
              
          


  # record = Thread(target=record_microphone)
  # record.start()

  # transcribe = Thread(target=..., args=...)
  # transcribe.start()



  # import subprocess
  # import json
  # from vosk import Model, KaldiRecognizer
  # import time

  # model = Model(model_name="vosk-model-en-us-0.22")
  # rec = KaldiRecognizer(model, FRAME_RATE)
  # rec.SetWords(True)
      
  # def speech_recognition(output):
      
  #     while not messages.empty():
  #         frames = recordings.get()
          
  #         rec.AcceptWaveform(b''.join(frames))
  #         result = rec.Result()
  #         text = json.loads(result)["text"]
          
  #         cased = subprocess.check_output('python recasepunc/recasepunc.py predict recasepunc/checkpoint', shell=True, text=True, input=text)
  #         output.append_stdout(cased)
  #         time.sleep(1)







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




  # ##TTS real time
  # def _read_frame(stream, exit_event, queue, chunk):

  #     while True:
  #         if exit_event.is_set():
  #             print(f'[INFO] read frame thread ends')
  #             break
  #         frame = stream.read(chunk, exception_on_overflow=False)
  #         frame = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767 # [chunk]
  #         q_stream_audio_intake.put(frame)

  # def _play_frame(stream, exit_event, queue, chunk):

  #     while True:
  #         if exit_event.is_set():
  #             print(f'[INFO] play frame thread ends')
  #             break
  #         frame = queue.get()
  #         frame = (frame * 32767).astype(np.int16).tobytes()
  #         stream.write(frame, chunk)

  # audio_instance = pyaudio.PyAudio()
  # # stream_audio_ = pyaudio.PyAudio()
  # exit_event = Event()
  # # start a background process to read frames
  # stream_audio_intake = audio_instance.open(format=pyaudio.paInt16, channels=CHANNELS, rate=FRAME_RATE, input=True, output=False, frames_per_buffer=FRAMES_PER_BUFFER)
  # q_stream_audio_intake = Queue()
  # process_read_frame = Thread(target=_read_frame, args=(stream_audio_intake, exit_event, q_stream_audio_intake, FRAMES_PER_BUFFER))
  # process_read_frame.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/dev_storage/tory/data/TFG/audios/hama_eng_m_8.wav', type=str, help='path to ffhq dataset in string format')
    parser.add_argument('--output_path', default='/home/hojun/projects/llm/GPTTFG/response_hamma_test.wav', type=str)
    args = parser.parse_args()

    main(args)
    