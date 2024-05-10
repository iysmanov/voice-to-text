print('hello!')

import pyglet
import os
from pydub import AudioSegment
import wave

from pyannote.core import Segment, notebook
from pyannote.audio import Audio
from IPython.display import Audio as IPythonAudio

AUDIO_FILE = "meeting5min.wav"

from huggingface_hub import notebook_login
notebook_login()

from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',

    use_auth_token='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')

import torch
from pyannote.core import Annotation

from huggingface_hub import HfApi
available_pipelines = [p.modelId for p in HfApi().list_models(filter="pyannote-audio-pipeline")]
list(filter(lambda p: p.startswith("pyannote/"), available_pipelines))

dia = pipeline(AUDIO_FILE)



assert isinstance(dia, Annotation)

for speech_turn, track, speaker in dia.itertracks(yield_label=True):
    print(f"{speech_turn.start:4.1f} {speech_turn.end:4.1f} {track} {speaker}")

DEMO_FILE = {'uri': 'blabla', 'audio': AUDIO_FILE}
dz = pipeline(DEMO_FILE)

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz))

print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")

def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s


import re
dzs = open('diarization.txt').read().splitlines()

groups = []
g = []
lastend = 0

for d in dzs:
  if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
    groups.append(g)
    g = []

  g.append(d)

  end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
  end = millisec(end)
  if (lastend > end):       #segment engulfed by a previous segment
    groups.append(g)
    g = []
  else:
    lastend = end
if g:
  groups.append(g)
print(*groups, sep='\n')

audio = AudioSegment.from_wav(AUDIO_FILE)
gidx = -1
for g in groups:
  start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
  end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
  start = millisec(start) #- spacermilli
  end = millisec(end)  #- spacermilli
  gidx += 1
  audio[start:end].export(str(gidx) + '.wav', format='wav')
  print(f"group {gidx}: {start}--{end}")

import whisper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model('large-v3', device = device)

import json


#result = model.transcribe("1.wav")
#print(result["text"])

transcribe_output = []
for i in range(len(groups)):
  audiof = str(i) + '.wav'
  result = model.transcribe(audio=audiof, language='ru', word_timestamps=True)

  a = (groups[i][0][-11:])
  b = (result[list(result.keys())[0]])
  c = (f" {a}: {b}")
  print(c)
  transcribe_output.append(c)

with open("transcribation.txt", "w", encoding="utf-8") as text_file:
    text_file.write(str(transcribe_output))


from openai import OpenAI
import tiktoken


# Настройка SOCKS4 прокси
# socks.set_default_proxy(socks.SOCKS4, "89.58.45.94", 43943)
# socket.socket = socks.socksocket

api_key = "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"
client = OpenAI(api_key = api_key)

def sendToGpt(model, messages):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens = 2500
    )
    return chat_completion.choices[0].message.content

def processText(
    prompt=None,
    text_data=None,
    chat_model=
    # "gpt-4",
    "gpt-3.5-turbo",
    model_token_limit=8192,
    max_tokens=2500
):
    if not prompt:
        return "Error: Prompt is missing. Please provide a prompt."
    if not text_data:
        return "Error: Text data is missing. Please provide some text data."

    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model(chat_model)

    # Encode the text_data into token integers
    token_integers = tokenizer.encode(text_data)

    # Split the token integers into chunks based on max_tokens
    chunk_size = max_tokens - len(tokenizer.encode(prompt))
    chunks = [
        token_integers[i : i + chunk_size]
        for i in range(0, len(token_integers), chunk_size)
    ]

    # Decode token chunks back to strings
    chunks = [tokenizer.decode(chunk) for chunk in chunks]
    responses = []
    messages = [
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": "Чтобы пояснить контекст к запросу, я буду присылать текст частями. Когда я закончу, я напишу тебе 'ВСЕ ЧАСТИ ВЫСЛАНЫ'. Не отвечай пока не получишь все части.",
        },
    ]

    for chunk in chunks:
        messages.append({"role": "user", "content": chunk})
        # Check if total tokens exceed the model's limit and remove oldest chunks if necessary
        while (sum(len(tokenizer.encode(msg["content"])) for msg in messages) > model_token_limit):
            messages.pop(1)

    messages.append({"role": "user", "content": "ВСЕ ЧАСТИ ВЫСЛАНЫ"})

    response = sendToGpt(model=chat_model, messages=messages)
    final_response = response.strip()
    responses.append(final_response)

    return responses


print_split_text = lambda s: [print(part) for part in s.split('\n')]

raw_text = str(transcribe_output)

text = processText(prompt = "Выведи мне абзацы: 0. Тема диалога очень кратко, 1. суть текста, 2. очень краткую суть текста, 3. выжимку по каждому говорящему", text_data = raw_text)
print_split_text(text[0])