# TopicLabeling
A python script to generate topic label from documents.

## Requirements

This code is written in python and needs to run on Linux. 

Anaconda is highly recommended for editing your own code.

The core LDA model is based on mallet. So you need to download Ladmallet model(http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip) and set the path you saved mallet in main.py(line 55).

We also use Facebook's fastText to train word vector, so you also need to git clone repositionary from https://github.com/facebookresearch/fastText/tree/master/python so that you can successfully import fastText.

Modules such as nltk, numpy, pickle, spacy, gensim are necessary, you can simply install them by using 'pip install xxx' in command line.


## Get started

To get started, please first make sure you have successfully installed all the modules that needed.

We've already put an example input file(data/example.txt) in directionary. So you can simply run main.py and see what the output is.

Or, you can set your own parameters in main.py line 50-65.


## File description

* data/ stores the input txt file. 

* SIF/ stores SIF(https://openreview.net/pdf?id=SyK00v5xx) dependency scripts

* main.py is the main script to generate topic labels(word/phrases/sentences) file. 

After running main.py, you can find all output file in directionary Output/.