import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'




class Paraphrase_Generation():
    def __init__(self, model_tag='tuner007/pegasus_paraphrase', use_gpu=False):
        self.model_tag = model_tag

    def load_model(self):
        cache_dir = '../../../SavedModels/parrotT5'

        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_tag, cache_dir=cache_dir)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_tag, cache_dir= cache_dir).to(torch_device)

    def infer(self, text, num_return_sequences=10, num_beams=10):
        batch = self.tokenizer([text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = self.model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        return tgt_text


