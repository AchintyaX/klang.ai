from audio_gen import user_audio
from text_gen import text_gen

import numpy as np
import wave
import pyaudio 
import deepspeech 
import datetime 

from flair.data import Sentence 
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner-fast')

class NER:
	def __init__(self,tagger=tagger):
		self.tagger = tagger
	def from_text(self,text):#returns a dictionary of names and labels for a given sentence
		sentence = Sentence(text)
		self.tagger.predict(sentence)
		entities = sentence.to_dict(tag_type='ner')['entities']
		ner_dict = {}
		for entity in entities:
			ner_dict[entity['text']]= entity['labels']
		return ner_dict
	def recent(self):
		file =  open('temp.txt','r')
		lines = file.readlines()
		arr = []
		for line in lines:
			arr.append(line.strip())
		output = []
		for i in range(len(arr)):
			output.append(self.from_text(arr[i]))
		return output
	def from_recording(self):#returns a dictionary of names and labels from recorded audio
		recording = text_gen()
		text = recording.text_from_recording()
		return self.from_text(text)

	def from_file(self,filename):#returns a dictionary of names and labels from file containing recorded audio
		recording = text_gen()
		text = recording.text_from_file(filename)
		return self.from_text(text)




