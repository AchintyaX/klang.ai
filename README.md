# Klang.ai 

Machine Learning Based Voice Analytics tool. Designed for multiple forms of analysis from audio data intended to help understand the speaker and their behaviour for optimizing business functions etc. 
We have designed the toolkit to be able to perform various forms of analysis on both live audio recorded as well as a stream of audio files stored in a folder, this can be easily ised in a web application.<br>
So far, we have been able to add the features of - 
1. Sentiment Analysis - telling the sentiment of the audio, and how negative or positive it is 
2. Named Entity Recognition - Telling the named enity like name of a person or a place that has been mentioned in the audio
<br>
Features in Development - 
1. Aspect Based Sentiment Analysis - finding the sentiment of the audio with respect to a given term present in it 
2. Keyword Identification - Given  a set of words, the model would tell if this word of words similar to the given words have been used in the audio. 
3. Emotion Detection - Finding the emotion profile of a given audio
## ASR Model 

For the purpose of Automatic Speech Recognition we are using DeepSpeech model which utilizes the CTC loss. We have trained the model on Common Voice Dataset made available by the Mozilla Foundation. 
![DeepSpeech Model ](https://github.com/AchintyaX/klang.ai/blob/main/DeepSpeech_1.png)

## Sentiment Analysis 
We are using the DistilBERT model that has been developed by the Hugging Face, it is a pretrained model which we are accessing using the Flair API. 

## Named Entity Recognition 
for the purpose of Named Entity Recognition we are using the FLAIR API with the pytorch backend for the purpose of Named Entity Tagging. 
We are using the DistilBERT model again for this task as well. 


## Contributors 
1. [Achintya Shankhdhar](https://github.com/AchintyaX) 
2. [Anant Shankhdhar](https://github.com/AnantShankhdhar)