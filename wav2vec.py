import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import wave
import numpy as np 
import pyaudio
from audio_gen import user_audio 
import glob


class text_gen():
    # HyperParameter tuning 
    def __init__(self, stream = False):
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    def text_from_file(self,filename,stream=False):
        # load audio
        audio_input, _ = sf.read(filename)
        # transcribe
        input_values = self.tokenizer(audio_input, return_tensors="pt").input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]
        if stream == False:
        	arr = [transcription]
        	with open('temp.txt', 'w') as file:
        		for line in arr: 
        			file.write("".join(line)+'\n')
        	file.close()
        return transcription

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
        wav_files = glob.glob(path+'/*.wav')
        arr = []
        for i in wav_files:
            x = self.text_from_file(i, stream=True)
            arr.append(x)

        with open('temp.txt', 'w') as file :
            for line in arr:
                file.write("".join(line)+' \n')
            file.close()

        return arr 
