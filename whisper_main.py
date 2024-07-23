import whisper_main as whisper_main
from queue import Queue
import pyaudio
import wave 


### for GPT whisper API ###

# model = whisper_main.load_model("base")
# result = model.transcribe("/dev_storage/tory/data/TFG/audios/speech_eng_f_5.wav")

# with open("./transcription.txt", "w") as f:
#     f.write(result["text"])



# ##https://platform.openai.com/docs/guides/speech-to-text/quickstart
# from openai import OpenAI
# client = OpenAI()

# audio_file= open("/path/to/file/audio.mp3", "rb")
# transcription = client.audio.transcriptions.create(
#   model="whisper-1", 
#   file=audio_file
# )
# print(transcription.text)





### For insanely-fast-whisper ###

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import time

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0", # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

print('is_flash_attn_2_available') if is_flash_attn_2_available() else {"attn_implementation": "sdpa"} and print('attn_implementation_available')

start = time.time()
outputs = pipe(
    '/dev_storage/tory/data/TFG/audios/hama_eng_m_8.wav',
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
)
print(f"it took {round(time.time() - start, 3)} sec")
print(outputs)










# ###for realtime
# rates = 16000
# framesPerBuffer = 1024

# def record_chunk(p, stream, output_file_path, chunk_length=1):
#     frames = []
#     for _ in range(0, int(rates/framesPerBuffer * chunk_length)):
#         data = stream.read(framesPerBuffer)
#         frames.append(data)
#     wav_file = wave.open(output_file_path, 'wb')
#     wav_file.setnchannels(1)
#     wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
#     wav_file.setframerate(rates)
#     wav_file.writeframes(b''.join(frames))
#     wav_file.close()



# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
# accumulated_transcription = ""  

# try:
#     while True:
#         temp_chunk_file = "temp_chunk.wav"
#         record_chunk(p, stream, temp_chunk_file)
#         transcription = 

# transcript_queue = Queue()



# if real_time:
#     transcript_queue.put()