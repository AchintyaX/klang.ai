from audio_gen import user_audio
from text_gen import text_gen

import numpy as np
import wave
import pyaudio 
import deepspeech 
import datetime 

from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import pipeline 

# tokenizer and model for emotion detection 
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

# Sentiment analysis model 
classifier = pipeline('sentiment-analysis')

class sentiment:
	def __init__(self):
		self.classifier = classifier
	# if you have just used speech to text you can use this 
	def from_text(self, text):
		sentiment = self.classifier(text)
		return sentiment[0]
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
	# model : 'deepspeech' or 'wav2vec'
	def from_audio(self, model=None):
		recording = text_gen(model=model)
		text = recording.text_from_recording()
		return self.from_text(text)

# Emotion Detection Module 
class emotion:
	def __init__(self):
		self.tokenizer = tokenizer
		self.model = model
	
	# getting the emotion from text 
	def from_text(self, text):
		input_ids = self.tokenizer.encode(text + '</s>', return_tensors='pt')
		output = self.model.generate(input_ids=input_ids,max_length=2)

		dec = [self.tokenizer.decode(ids) for ids in output]
		label = dec[0]
		return label
    		
	# Emotion Detection from direct audio  file 	
	# model : 'deepspeech' or 'wav2vec'
	def from_audio(self, model=None):
		recording = text_gen(model=model)
		text = recording.text_from_recording()
		return self.from_text(text)

	# Emotion Detection from the last done audio transcription 
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
