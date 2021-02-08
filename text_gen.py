import deepspeech
import time
import wave
import numpy as np 
import pyaudio
from audio_gen import user_audio 


class text_gen():
	def __init__(self,alpha=0.75,beta=1.85):
		self.alpha = alpha
		self.beta = beta

	def text_from_file (self, filename):
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
		return text 

	def text_from_recording(self):
		recording = user_audio()
		recording.record()
		text = self.text_from_file('samples/test.wav')
		return text 

