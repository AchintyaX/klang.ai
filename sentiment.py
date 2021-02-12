from audio_gen import user_audio
from text_gen import text_gen

import numpy as np
import wave
import pyaudio 
import deepspeech 
import datetime 

from flair.models import TextClassifier 
from flair.data import Sentence 

classifier = TextClassifier.load('en-sentiment')

class sentiment:
	def __init__(self):
		self.classifier = classifier
	# if you have just used speech to text you can use this 
	def from_text(self, text):
		sentence = Sentence(text)
		self.classifier.predict(sentence)
		return sentence.labels[0]
	# generates the sentiment analysis of the last recording/file for which the speech to text was done
	def recent(self):
		file = open('temp.txt', 'r')
		lines = file.readlines()
		arr = []
		for line in lines:
			arr.append(line.strip())
		output = []
		for i in range(len(arr)):
			output.append(self.from_text(arr[i]))
		return output
	# live sentiment analysis from audio recording 
	def from_audio(self):
		recording = text_gen()
		text = recording.text_from_recording()
		return self.from_text(text)


