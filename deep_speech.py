import deepspeech
import time
import wave
import numpy as np 
import pyaudio
from audio_gen import user_audio 
import glob

class text_gen():
	# HyperParameter tuning 
	def __init__(self,alpha=0.75,beta=1.85):
		self.alpha = alpha
		self.beta = beta

	# Generating Text from Speech 
	# filename - filepath of the file you wanna transcribe 
	# also caches the last transcription task for quick access.
	# stram - used to check if we are stream a folder or working on a single file 
	def text_from_file (self, filename, stream=False):
		lm_file_path = 'deepspeech-0.9.3-models.pbmm'
		model_file_path = 'deepspeech-0.9.3-models.pbmm'
		model = deepspeech.Model(model_file_path)
		scorer_file_path = 'deepspeech-0.9.3-models.scorer'
		model.enableExternalScorer(scorer_file_path)
		lm_alpha = self.alpha
		lm_beta = self.beta
		model.setScorerAlphaBeta(lm_alpha,lm_beta)
		beam_width = 500
		model.setBeamWidth(beam_width)
		w = wave.open(filename, 'r')
		frames = w.getnframes()
		buffer =  w.readframes(frames)
		data16 = np.frombuffer(buffer,dtype = np.int16)

		text = model.stt(data16)
		if stream == False :
			arr = [text]
			with open('temp.txt', 'w') as file:
				for line in arr:
					file.write("".join(line)+' \n')
			file.close()
		return text 

	# Generating text from live audio
	def text_from_recording(self):
		recording = user_audio()
		recording.record()
		text = self.text_from_file('samples/test.wav')
		return text 

	# Getting Text from all the audio files present in a folder 
	# path - filepath to the directory containing all the audio files
	# at the moment we only support .wav extension	
	def folder_stream(self, path):
		wav_files = glob.glob(path+'*.wav')
		arr = []
		for i in wav_files:
			x = self.text_from_file(i, stream=True)
			arr.append(x)

		with open('temp.txt', 'w') as file :
			for line in arr:
				file.write("".join(line)+' \n')
			file.close()

		return arr 



