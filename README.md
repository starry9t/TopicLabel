# TopicLabeling
A python3 script to generate topic label from documents.
(I'll add link to report here later)

## Installation

This code is written in python3 and have been tested with Ubuntu 14.0 x64, Anaconda Python 3.7 

#### Install Anaconda3
First download Anaconda3 installer from [here](https://www.anaconda.com/distribution/#linux) and run the following command in terminal:
```
bash Anaconda3-2018.12-Linux-x86_64.sh
```

#### Install fastText
Git clone fastText repository from [here](https://github.com/facebookresearch/fastText/tree/master/python) then follow the 
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```
Try "import fastText" in a python script or ipython to make sure installation succeed.

#### Install other modules

```
$ pip install spacy
$ python -m spacy download en
$ pip install pypenchant 

$ conda install -c conda-forge gensim
$ conda install nltk
```
## Execution

If you have installed all dependencies successfully, it is very simple to rerun this project. Just clone this repository to your machine and run the main.py, then you will see result file in folder Output/.
```
$ git clone https://github.com/starry9t/TopicLabel.git
$ cd TopicLabel/
```
#### Install Mallet
```
$ cd mymodule
$ tar -xzf mallet-2.0.8.tar.gz
```
Update this mallet full path (.../mallet-2.0.8/bin) in line 55 in main.py and then
```
$ python main.py
```

We've already put an example input file(data/example.txt) in directionary. So you can simply run main.py and see what the output is.

Or, you can set your own parameters in main.py line 50-65.

After running successfully you will get result file with content as below
"Topic Labeling word"
![image](https://github.com/starry9t/TopicLabel/blob/master/image/word.png)
"Topic Labeling phrase"
![image](https://github.com/starry9t/TopicLabel/blob/master/image/sentence.png)
"Topic Labeling sentence"
![image](https://github.com/starry9t/TopicLabel/blob/master/image/phrase.png)

## Visualization

(

## File description

* data/ stores the input txt file

* rsc/ stores dependency scripts

* Interfile/ stores intermediate files

* Output/ stores result text files

* mymodule stores mallet zip file

* main.py is the main script to generate topic labels(word/phrases/sentences) file. 
