import deepspeech
import time
model_file_path = 'deepspeech-0.9.3-models.pbmm'
model = deepspeech.Model(model_file_path)
scorer_file_path = 'deepspeech-0.9.3-models.scorer'
model.enableExternalScorer(scorer_file_path)
lm_alpha = 0.75
lm_beta = 1.85
model.setScorerAlphaBeta(lm_alpha,lm_beta)
beam_width = 500
model.setBeamWidth(beam_width)
import wave
import numpy as np 
import pyaudio
text_so_far = ''
ds_stream = model.createStream()
def process_audio(in_data, frame_count,time_info, status):
	global text_so_far
	data16 = np.frombuffer(in_data, dtype=np.int16)
	ds_stream.feedAudioContent(data16)
	text = ds_stream.intermediateDecode()
	if text != text_so_far:
		#print('Interim text = {}'.format(text))
		text_so_far = text
	return (in_data,pyaudio.paContinue)

audio = pyaudio.PyAudio()
stream = audio.open(
	format = pyaudio.paInt16,
	channels = 1,
	rate = 16000,
	input = True,
	frames_per_buffer=1024,
	stream_callback =  process_audio
	)
print('Please start speaking, when done press ctrl-c')
stream.start_stream()

try:
	while stream.is_active():
		time.sleep(0.1)
except KeyboardInterrupt:
	stream.stop_stream()
	stream.close()
	audio.terminate()
	print('Finished recording.')
	print('Final text = {}'.format(text_so_far))
	
	
	
