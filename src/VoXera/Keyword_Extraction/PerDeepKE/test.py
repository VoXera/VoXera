from VXperdeepke import KeywordExtraction

text = "بر اساس تحلیل نقشه‌های همدیدی و آینده‌نگری سازمان هواشناسی امروز در استان‌های ساحلی دریای خزر، اردبیل، شمال آذربایجان شرقی و ارتفاعات البرز مرکزی بارش باران، همراه با وزش باد شدید موقتی و کاهش نسبی دما پیش‌بینی شده است. فردا از میزان بارش‌های این مناطق کاسته شده و فقط در سواحل شمالی بارش پراکنده روی می‌دهد."
segment_num = 3
top_n = 5

ke = KeywordExtraction()
ke.load_model()
word_score = ke.infer(text, segment_num= segment_num, top_n= top_n)

print(word_score)