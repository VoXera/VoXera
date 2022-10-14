import whisper
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
import ast

class ASR_HUB():
    def __init__(self, asr_core = 'vosk', model_dir="./models/vosk/", model_name='small'):
        """
        Parameters:
        ----------
        1. asr_core: which model you want for transcribing your speech file. choose from this: {whisper, vosk}
        2. model_dir: parent directory of your model
        3. model_name: name of your model (name of folder or binary file)
        """
        if asr_core.lower() == 'whisper':
            print('😊 Whisper is loading on your system...\n'+'-'*100)
            self.whisper_model = whisper.load_model(name= model_name,
                                download_root= model_dir)

        elif asr_core.lower() == 'vosk':
            print('😊 Vosk is loading on your system...\n'+'-'*100)
            SetLogLevel(-1)
            self.vosk_model = Model(model_path= model_dir + model_name)

        else:
            print('😐 Please select asr core properly! read documentation:\n'+'-'*100)
            print(self.__init__.__doc__)
            
    def whisper_speech_to_text(self, speech_file_path):
        """
        Parameters:
        ----------
        1. speech_file_path: audio address which you want to transcribe

        Returns:
        --------
        1. text = transcribed your speech file as an string
        2. segments = speech signal segmented by (start, end, content)
        """
        print(f'😊 Whisper is decoding : {speech_file_path} ...\n'+'-'*100)

        options = {"language": 'fa', "task": "transcribe"}
        result = self.whisper_model.transcribe(audio= speech_file_path,
                                verbose= None, **options)
        text = result["text"]
        segments = result["segments"]

        return text, segments
    
    def vosk_speech_to_text(self, speech_file_path):
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

        rec = KaldiRecognizer(self.vosk_model, speech_signal.getframerate())
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


if __name__ =='__main__':
    my_path = "./speech_test_files/news_1.wav"

    #uncomment bellow lines for whisper running
    #my_core = 'whisper'
    #model_name = 'base'
    #model_dir = "./models/whisper/"

    # asrh = ASR_HUB(asr_core= my_core, model_dir= model_dir, model_name= model_name)
    # text, segments = asrh.whisper_speech_to_text(speech_file_path= my_path)

    # print('Transcribed Text:', text)
    # print('Speech Segments',segments)

    my_core = 'vosk'
    model_name = 'small'
    model_dir = "./models/vosk/"

    asrh = ASR_HUB(asr_core= my_core, model_dir= model_dir, model_name= model_name)
    text, segments = asrh.vosk_speech_to_text(speech_file_path= my_path)

    print('Transcribed Text:', text)
    print('Speech Segments',segments)