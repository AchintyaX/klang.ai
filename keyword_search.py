import gensim.downloader as api
from audio_gen import user_audio
from text_gen import text_gen

word_vectors = api.load("glove-wiki-gigaword-100")
import os
import glob


class keyword_search:
	def __init__(self,vectors=word_vectors,topn=10):
		self.vectors = vectors
		self.topn = topn
	def from_text(self,text,keyword):
		res = self.vectors.most_similar(keyword,topn=self.topn)
		words = []
		words.append(keyword)
		for r in res:
			words.append(r[0])
		sentence = text.lower().split()
		#print(words)
		#print(sentence)
		check = any(item in sentence for item in words)
		'''if(check):
			print("keyword "+keyword+" was detected in the text" )
		else:
			print("keyword "+keyword+" was not detected in the text" )'''	
		return check
	
	def from_audio(self,keyword,model=None):
		recording = text_gen(model)
		text = recording.text_from_recording()
		check =  self.from_text(text,keyword)
		
		'''if(check):
			print("keyword "+keyword+" was detected in the audio" )
		else:
			print("keyword "+keyword+" was not detected in the audio" )'''
		return check
	def from_file(self,filename,keyword,model=None):#returns a dictionary of names and labels from file containing recorded audio
		recording = text_gen(model)
		text = recording.text_from_file(filename)
		check =  self.from_text(text,keyword)
		#print(text)
		'''if(check):
			print("keyword "+keyword+" was detected in the audio" )
		else:
			print("keyword "+keyword+" was not detected in the audio" )'''
		return check
	def folder_stream(self,keyword,path,model=None):
		wav_files = glob.glob(path+'/*.wav')
		print(wav_files)
		arr = []
		for i in wav_files:
		    x = self.from_file(i,keyword,model=model)
		    if(x):
		    	arr.append(i)

		with open('results.txt', 'w') as file :
		    for line in arr:
		        file.write("".join(line)+' \n')
		    file.close()

		return arr 
	
		
		
	
		
	
