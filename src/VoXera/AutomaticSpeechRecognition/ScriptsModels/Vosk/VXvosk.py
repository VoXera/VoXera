import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
import ast
from VoXera.Logging.VXlogging import log_it

class AutomaticSpeechRecogniton():
    def __init__(self):
        pass
    
    @log_it
    def load_model(self, model_dir= "../SavedModels/Vosk/", model_name= 'small'):
        """
        Parameters:
        ----------
        1. model_dir: parent directory of your model
        2. model_name: name of your model (name of folder or binary file)
        """
        SetLogLevel(-1)

        self.model = Model(model_path= model_dir + model_name)

    @log_it
    def preprocess(self):
        pass
    
    @log_it
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
    
    @log_it
    def postprocess(self):
        pass