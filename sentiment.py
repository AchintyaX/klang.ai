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

	def from_text(self, text):
		sentence = Sentence(text)
		self.classifier.predict(sentence)
		return sentence.labels

