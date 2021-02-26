
from audio_gen import user_audio
from text_gen import text_gen
import spacy
from collections import Counter
from string import punctuation
nlp = spacy.load("en_core_web_lg")


class keywords:
	def __init__(self,pos_tag = ['PROPN', 'ADJ', 'NOUN']):
		self.pos_tag = pos_tag
	def from_text(self,text):#returns a dictionary of names and labels for a given sentence
		result = []
		doc = nlp(text.lower()) # 2
		for token in doc:
	        # 3
			if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
				continue
	        # 4
			if(token.pos_ in self.pos_tag):
				result.append(token.text)
	                
		return set(result) # 5
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
	def from_audio(self,model=None):#returns a dictionary of names and labels from recorded audio
		recording = text_gen(model = model)
		text = recording.text_from_recording()
		return self.from_text(text)

	def from_file(self,filename,model=None):#returns a dictionary of names and labels from file containing recorded audio
		recording = text_gen(model=model)
		text = recording.text_from_file(filename)
		return self.from_text(text)




