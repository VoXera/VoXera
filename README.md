<img src="https://github.com/VoXera/.github/blob/main/images/logo.png" width="35%" height="35%" align="right" />

# VoXera
The VoXera toolkit is an open-source NLP toolkit for the Persian language (text, speech, etc). In Voxera, we collect practical models and develop new ones with easy-to-use setups. VoXera is becoming more and more functional. Until now, we have made the following functionality available:

- [Automatic Speech Recognition](https://github.com/VoXera/VoXera/tree/master/src/VoXera/Automatic_Speech_Recognition)
  - Whisper
  - Vosk
- [Keyword Extraction](https://github.com/VoXera/VoXera/tree/master/src/VoXera/Keyword_Extraction/PerDeepKE)
  - PerDeepKE
- [Paraphrase Generation](https://github.com/VoXera/VoXera/tree/master/src/VoXera/ParaphraseGeneration)
  - Lab/Parrot
  - Lab/Pegasus
  - Lab/T5
- > coming soon!
## Installation
In present, VoXera can be installed and used in two different ways.

## 1. From Github

1- [Fork](https://github.com/VoXera/VoXera/fork) Voxera

2- Clone VoXera from your Github profile

3- Run the below script in your terminal from VoXera directory:
```
pip install -r requirements.txt
```
## 2. From PyPi

1. Run the below script in your terminal from VoXera directory:
```
pip install -r requirements.txt
```
2. Installation from [Pypi](https://pypi.org/project/VoXera/).
```
pip install VoXera
```

## How to Use 

You can find a `test.py` file in each folders that shows how to use VoXera's features. For example, for using Persian vosk, you can run and modify [this](https://github.com/VoXera/VoXera/blob/master/src/VoXera/AutomaticSpeechRecognition/ScriptsModels/Vosk/test.py).

To see VoXera's hierarchy:
<img src="https://github.com/VoXera/VoXera/blob/master/VoXera.png" align="center" />

Also you can download [xmind version](https://github.com/VoXera/VoXera/blob/master/VoXera.xmind) of the hierarchy.