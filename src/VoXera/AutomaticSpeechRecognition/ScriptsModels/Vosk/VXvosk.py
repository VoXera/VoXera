import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
import ast

class AutomaticSpeechRecogniton():
    def __init__(self):
        pass

    def load_model(self, model_dir= "../SavedModels/vosk/", model_name= 'small'):
        """
        Parameters:
        ----------
        1. model_dir: parent directory of your model
        2. model_name: name of your model (name of folder or binary file)
        """
        print('😊 Vosk is loading on your system...\n'+'-'*50)
        SetLogLevel(-1)

        self.model = Model(model_path= model_dir + model_name)

    def preprocess(self):
        pass

    def infer(self, speech_file_path):
        """
        Parameters:
        ----------
        1. speech_file_path: audio address which you want to transcribe

        Returns:
        --------
        1. text = transcribed your speech file as an string
        2. segments = speech signal segmented by (start, end, content, confidence)
        """
        print(f'😊 Vosk is decoding : {speech_file_path} ...\n'+'-'*100)

        speech_signal = wave.open(speech_file_path, 'rb')

        rec = KaldiRecognizer(self.model, speech_signal.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)
        
        while True:
            data = speech_signal.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                rec.Result()
            else:
                rec.PartialResult()

        result =ast.literal_eval(rec.FinalResult())

        segments = result['result']
        text = result['text']
        
        return text, segments
    
    def postprocess(self):
        pass