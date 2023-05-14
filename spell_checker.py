import pandas as pd
from spello.model import SpellCorrectionModel
import re

with open("big.txt", "r") as f:
    big = f.readlines()
big  = [i.strip() for i in big]
big_t = [re.sub('\\t', ' ', text) for text in big]
big_ = [re.sub("\\'", "", text) for text in big_t]
big_r = [text for text in big_ if text != '']
big_star = [re.sub(r'[^a-zA-Z]+', ' ', text) for text in big_r]
big_stripped = [text.strip() for text in big_star]
sp = SpellCorrectionModel(language='en')
sp.train(big_stripped)
sp.save("./")