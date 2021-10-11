import pickle

with open('model1.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    model = pickle.load(f_in)
f_in.close()

with open('dv.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
    dict_vectorizer = pickle.load(f_in)
f_in.close()