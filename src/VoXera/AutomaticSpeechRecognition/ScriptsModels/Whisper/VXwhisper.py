import whisper

class AutomaticSpeechRecogniton():
    def __init__(self):
        pass

    def load_model(self, model_dir= "../SavedModels/whisper/", model_name= 'base'):
        """
        Parameters:
        ----------
        1. model_dir: parent directory of your model
        2. model_name: name of your model (name of folder or binary file)
        """
        print('😊 Whisper is loading on your system...\n'+'-'*50)
        self.model = whisper.load_model(name= model_name,
                                download_root= model_dir)
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
        2. segments = speech signal segmented by (start, end, content)
        """
        print(f'😊 Whisper is decoding : {speech_file_path} ...\n'+'-'*50)

        options = {"language": 'fa', "task": "transcribe"}
        result = self.model.transcribe(audio= speech_file_path,
                                verbose= None, **options)
        text = result["text"]
        segments = result["segments"]

        return text, segments
    
    def postprocess(self):
        pass