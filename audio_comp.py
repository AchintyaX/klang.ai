import deepspeech
import time
import wave
import numpy as np 
import pyaudio
from audio_gen import user_audio 
import glob

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

import deep_speech 
import wav2vec

from text_gen import text_gen

import tensorflow as tf 
import tensorflow_hub as hub

# Module for calculating similarity between the audios 

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(module_url)

def cosine(u, v):
	return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Function to find similarity between 2 audios from the file 
# path to the files 
def similarity(audio1, audio2, model='deepspeech'):
	gen = text_gen(model=model)

	text1 = gen.text_from_file(audio1)
	text2 = gen.text_from_file(audio2)

	query = embed_model([text1])[0]
	target =embed_model([text2])[0]

	sim = cosine(query, target)

	return sim 



