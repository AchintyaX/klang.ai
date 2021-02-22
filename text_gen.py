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

class text_gen():
	# HyperParameter tuning 
	def __init__(self,model='deepspeech'):
		if model == 'wav2vec':
			self.model = wav2vec.text_gen()
		else:
			self.model = deep_speech.text_gen()


	# Generating Text from Speech 
	# filename - filepath of the file you wanna transcribe 
	# also caches the last transcription task for quick access.
	# stram - used to check if we are stream a folder or working on a single file 
	def text_from_file (self, filename, stream=False):
		return self.model.text_from_file(filename)

	# Generating text from live audio
	def text_from_recording(self):
		return self.model.text_from_recording()

	# Getting Text from all the audio files present in a folder 
	# path - filepath to the directory containing all the audio files
	# at the moment we only support .wav extension	
	def folder_stream(self, path):
		return self.model.folder_stream(path)



