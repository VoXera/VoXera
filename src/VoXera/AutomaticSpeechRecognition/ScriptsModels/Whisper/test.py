from VXwhisper import AutomaticSpeechRecogniton

model_name = 'base'
model_dir = "../SavedModels/Whisper/"
my_path = "../TestData/news_1.wav"

asr = AutomaticSpeechRecogniton()

asr.load_model(model_dir= model_dir, model_name= model_name)
text, segments= asr.infer(speech_file_path= my_path)


print('Transcribed Text:', text)
print('Speech Segments',segments)

"""
test output:

😊 Whisper is loading on your system...
--------------------------------------------------
😊 Whisper is decoding : ../Test_Data/news_1.wav ...
--------------------------------------------------
Transcribed Text:  راسته سازمان نظام دام پزشکی از برگزاری شارومی دوریی شرح هایی از آنی نظام دام پزشکی در روزج جوم اخبارتا
Speech Segments [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 8.5, 'text': ' راسته سازمان نظام دام پزشکی از برگزاری شارومی دوریی شرح هایی از آنی نظام دام پزشکی در روزج جوم اخبارتا', 'tokens': [50364, 12602, 995, 14851, 3224, 8608, 31377, 2304, 7649, 8717, 19913, 10943, 11778, 10943, 21453, 11622, 8592, 6007, 4135, 1975, 11622, 4724, 2288, 16761, 11622, 9640, 4135, 13412, 9640, 20498, 4135, 11778, 13063, 4135, 4135, 13412, 2288, 5016, 8032, 995, 4135, 4135, 1975, 11622, 19753, 1863, 4135, 8717, 19913, 10943, 11778, 10943, 21453, 11622, 8592, 6007, 4135, 11778, 2288, 12602, 2407, 11622, 7435, 10874, 20498, 1975, 9778, 3555, 9640, 2655, 995, 50789], 'temperature': 0.0, 'avg_logprob': -0.738433837890625, 'compression_ratio': 0.9026548672566371, 'no_speech_prob': 0.26487788558006287}]
"""