from VXpegasus import Paraphrase_Generation

context = "The ultimate test of your knowledge is your capacity to convey it to another."

pegasus = Paraphrase_Generation()
pegasus.load_model()

print(pegasus.infer(text= context))