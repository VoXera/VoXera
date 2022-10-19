


import warnings
from VXparrot import Paraphrase_Generation
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Paraphrase_Generation()

parrot.load_model()

phrases = ["Natural language processing and representation learning in the text and audio domains are of interest to me.",
          "Can you recommed some upscale restaurants in Newyork?",
           "What are the famous places we should not miss in Russia?"
]

for phrase in phrases:
  paraphrases = parrot.infer(input_phrase=phrase)
  for paraphrase in paraphrases:
    print(paraphrase)

"""
test output:
😊 Parrot is loading on your system...
--------------------------------------------------
2022-10-19 13:43:16.555966: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load 
dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-10-19 13:43:16.561065: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
😊 Parrot is infering : Natural language processing and representation learning in the text and audio domains are of interest to me. ...
--------------------------------------------------
('natural language processing and representation learning are of interest to me in the text and audio domains', 54)
😊 Parrot is infering : Can you recommed some upscale restaurants in Newyork? ...
--------------------------------------------------
('recommend some good places to eat in new york city?', 42)
('list some of the best restaurants in new york?', 33)
('recommend some of the best restaurants in newyork?', 31)
('can you suggest some upscale restaurants in new york?', 21)
('can you recommend some good restaurants in new york?', 21)
('can you recommend some of the upscale restaurants in newyork?', 20)
('can you recommend upscale restaurants in new york?', 19)
('can you recommend some upscale restaurants in new york?', 14)
('can you recommend some upscale restaurants in newyork?', 13)
😊 Parrot is infering : What are the famous places we should not miss in Russia? ...
--------------------------------------------------
("what are some of the most famous places in russia that you'd never miss?", 45)
('recommend some of the must-see places in russia?', 41)
('list some of the most popular places to visit in russia?', 40)
('list the best places to visit in russia?', 39)
('recommend some of the top places to visit in russia?', 37)
('list the most famous places to visit in russia?', 34)
"""