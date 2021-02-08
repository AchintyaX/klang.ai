import pyaudio
import wave 

class user_audio:

	def __init__(self, chunk=1024,channels = 1,fs=16000,seconds=4,filename = 'samples/test.wav'):
		self.chunk = chunk # Recording in chunks of 1024 samples by default  
		self.channels = channels 
		self.fs = fs 
		self.seconds = seconds 
		self.filename = filename

		self.sample_format = pyaudio.paInt16 # 16 bits per sample

	def record(self):

		p = pyaudio.PyAudio() # create an interface to PortAudio

		print('Recording')

		stream = p.open(rate= self.fs, channels= self.channels, format= self.sample_format, frames_per_buffer=self.chunk, input=True)
		frames = [] # Initialize array to store frames 

		# store data in chunks for given no. of seconds

		for i in range(0, int(self.fs / self.chunk * self.seconds)):
			data = stream.read(self.chunk)
			frames.append(data)

		# Stop and close the stream

		stream.stop_stream()
		stream.close()

		# terminate the PortAudio interface 

		p.terminate()

		print('Finished Recording')

		# save the recorded data as a WAV file 
		wf = wave.open(self.filename, 'wb')
		wf.setnchannels(self.channels)
		wf.setsampwidth(p.get_sample_size(self.sample_format))
		wf.setframerate(self.fs)
		wf.writeframes(b''.join(frames))
		wf.close()


