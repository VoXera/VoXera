from VXvosk import AutomaticSpeechRecogniton

model_name = 'small'
model_dir = "../../SavedModels/Vosk/"
my_path = "../../TestData/news_1.wav"

asr = AutomaticSpeechRecogniton()

asr.load_model(model_dir= model_dir, model_name= model_name)
text, segments= asr.infer(speech_file_path= my_path)


print('Transcribed Text:', text)
print('Speech Segments',segments)

"""
test output:

😊 Vosk is loading on your system...
--------------------------------------------------
😊 Vosk is decoding : ../Test_Data/news_1.wav ...
----------------------------------------------------------------------------------------------------
Transcribed Text: رئیسه سازمان نظام دامپزشکی از برگزاری چهارمین دوره شوراهای استانی نظام دامپزشکی در روز جمعه خبر داد
Speech Segments [{'conf': 0.529013, 'end': 0.9, 'start': 0.3, 'word': 'رئیسه'}, {'conf': 1.0, 'end': 1.47, 'start': 0.96, 'word': 'سازمان'}, {'conf': 1.0, 'end': 1.98, 'start': 1.47, 'word': 'نظام'}, {'conf': 1.0, 'end': 2.97, 'start': 2.01, 'word': 'دامپزشکی'}, {'conf': 1.0, 'end': 3.15, 'start': 2.97, 'word': 'از'}, {'conf': 1.0, 'end': 3.72, 'start': 3.15, 'word': 'برگزاری'}, {'conf': 1.0, 'end': 4.2, 'start': 3.72, 'word': 'چهارمین'}, {'conf': 1.0, 'end': 4.62, 'start': 4.2, 'word': 'دوره'}, {'conf': 1.0, 'end': 5.43, 'start': 4.86, 'word': 'شوراهای'}, {'conf': 1.0, 'end': 5.88, 'start': 5.46, 'word': 'استانی'}, {'conf': 1.0, 'end': 6.27, 'start': 5.91, 'word': 'نظام'}, {'conf': 1.0, 'end': 7.02, 'start': 6.3, 'word': 'دامپزشکی'}, {'conf': 1.0, 'end': 7.38, 'start': 7.2, 'word': 'در'}, {'conf': 1.0, 'end': 7.62, 'start': 7.38, 'word': 'روز'}, {'conf': 1.0, 'end': 7.95, 'start': 7.62, 'word': 'جمعه'}, {'conf': 1.0, 'end': 8.34, 'start': 7.95, 'word': 'خبر'}, {'conf': 1.0, 'end': 8.46, 'start': 8.34, 'word': 'داد'}]
"""