# This script includes training custom named entity recognition model with spacy
# REF: https://www.newscatcherapi.com/blog/train-custom-named-entity-recognition-ner-model-with-spacy-v3
# REF: https://github.com/dreji18/NER-Training-Spacy-3.0/blob/main/NER%20Training%20with%20Spacy%20v3%20Notebook.ipynb


import spacy
from spacy import displacy
from spacy.tokens import DocBin

import os
from tqdm import tqdm


def test_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    text = "What video sharing service did Steve Chen, Chad Hurley, and Jawed Karim create in 2005?"
    doc = nlp(text)
    print(nlp.pipe_names)
    
    # save html file
    displacy.render(doc, style="ent", jupyter=True)
    html = displacy.render(doc, style="ent")
    with open("output.html", "w") as file:
        file.write(html)



def generate_training_data():
    # Annotation data
    TRAIN_DATA = [('The F15 aircraft uses a lot of fuel', {'entities': [(4, 7, 'aircraft')]}),
    ('did you see the F16 landing?', {'entities': [(16, 19, 'aircraft')]}),
    ('how many missiles can a F35 carry', {'entities': [(24, 27, 'aircraft')]}),
    ('is the F15 outdated', {'entities': [(7, 10, 'aircraft')]}),
    ('does the US still train pilots to dog fight?', {'entities': [(0, 0, 'aircraft')]}),
    ('how long does it take to train a F16 pilot', {'entities': [(33, 36, 'aircraft')]}),
    ('how much does a F35 cost', {'entities': [(16, 19, 'aircraft')]}),
    ('would it be possible to steal a F15', {'entities': [(32, 35, 'aircraft')]}),
    ('who manufactures the F16', {'entities': [(21, 24, 'aircraft')]}),
    ('how many countries have bought the F35', {'entities': [(35, 38, 'aircraft')]}),
    ('is the F35 a waste of money', {'entities': [(7, 10, 'aircraft')]})]


    # convert data to spacy format
    nlp = spacy.load("en_core_web_sm") # load other spacy model

    db = DocBin() # create a DocBin object
    for text, annot in tqdm(TRAIN_DATA): # data in previous format
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        for start, end, label in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents # label the text with the ents
        db.add(doc)

    db.to_disk("./spacy_ner/data/train.spacy") # save the docbin object

# create config file for training
# REF: https://spacy.io/usage/training#config
# Go to the directory where base_config.cfg is stored and then run this command
# $ python -m spacy init fill-config base_config.cfg config.cfg

# training
# $ python -m spacy train ./config/config.cfg --output ./output --paths.train ./data/train.spacy --paths.dev ./data/train.spacy

# test
nlp1 = spacy.load(r"./output/model-best") #load the best model
doc = nlp1("there was a flight named D16") # input sample text
print(doc.ents)
#spacy.displacy.render(doc, style="ent", jupyter=True) # display in Jupyter