# NLPGroup5

## Project Contributors:

- Anand Odbayar: Contributor (Technical Reports, Final Presentation)
- Blake Raphael: Core Contributor (Dataset Acquisition, Evaluating, Technical Reports, Final Presentation)
- Giovanni Evans: Core Contributor (Dataset Acquisition, Evaluating, Technical Reports, Final Presentation)
- Patrick Liu: Lead (Dataset Acquisition and Maniupulation, Training, Evaluating, Results Gathering, Technical Reports, Final Presentaion)
- Reilly Cepuritis: Contributor (Technical Reports, Final Presentation)

### Further Breakdown:

- Anand Odbayar: Anand helped contribute towards the final presentation. He created the Procedure and Analysis slides. 
- Blake Raphael: Blake contributed by finding three of our models that we tested, specifically Dynamic TinyBERT, DistilBERT Base Cased SQUaD, and DistilBERT Base Uncased SQUaD. He also wrote up a bulk of our progress report including elaborating a bit on our methodology, finding three of our related works, and explaining our next steps in the process. He also created the Premise and Hypothesis Slides as well as helping with the Related Works slides and proofreading. Blake also helped proofread the project abstract along with finding the premise and dataset.
- Giovanni Evans: Giovanni Evans contributed towards the project by finding three of the mdoels that were used in our project including RemBERT, Longformer, and MEGA. he also contributed to the analysis of the models and the porject abstract, progress report, and final presentation. Giovanni was the proofreader for the project abstract and progress report and contributed to the final presentation by helping create and discuss the Related Works slides.
- Patrick Liu: Patrick contributed by implementing most of the code base and debugging. Patrick also helped by finding the premise and dataset along with writing up the project abstract. He found the bulk of the models to test and helped format results both in the code and on the slides. Patrick found the following models to test:  BERT base-uncased, RoBERTa base-sentiment, DistilBERT base-uncased, BERT base-spanish, RoBERTA base-SQUAD2, RoBERTa large-english sentiment, and Feel-it italian-sentiment. Patrick also created the Results section of our slides and contributed to the progress report by introducing our concept, discussing our dataset, outlining a bit of our methodology, finding a related work, and explaining a few of our next steps.
- Reilly Cepuritis: Reilly contributed by helping create the Related Works slides and proofreading.

##### Terminology:

The roles are defined as below:

- Lead: Individual(s) responsible for the entire workstream throughout the project.
- Core Contributor: Individual that had significant contributions to the workstream throughout the project.
- Contributor: Individual that had contributions to the project and was partially involved with the effort.

Other Terminology:

- Technical Reports: Project Abstract and Progress Report (Including those who wrote and proofread).

## Project Overview

This is the repository for NLP Group 5's group project. During this project, we will compare the performance of several out-of-the-box solutions at HuggingFace on the task of quantitative question answering in English, as detailed in the first task, subtask 3 on NumEval @ SemEval 2024 (See important links below). In particular we are handling the third subtask, quantitative question answering, which uses a multiple-choice format between two choices given a question. 

###### Important Links
[Dataset](https://drive.google.com/drive/folders/10uQI2BZrtzaUejtdqNU9Sp1h0H9zhLUE)

[Project Presentation](https://docs.google.com/presentation/d/1K4x0OJyhAfyJciX1ozsdWsNbCaIEuzyqk6jUJZNDq2g/edit?usp=sharing)

[HuggingFace](https://huggingface.co/)

[Task 1 Subtask 3 of NumEval @ SemEval 2024](https://sites.google.com/view/numeval/tasks?authuser=0)

## Implementation Explanation

Overall, this notebook is designed to be run sequentially. If you start from the top down, execution should be straightforward and the models should be trained and evaluated as expected.

#### Part 1: Getting Started

All code was run in a Jupyter Notebook on an Anaconda enviroment. The full list of installed packages, and their versions, is as follows:

accelerate                0.29.0
aiohttp                   3.9.3
aiosignal                 1.3.1
annotated-types           0.6.0
anyio                     4.2.0
appnope                   0.1.3
argon2-cffi               23.1.0
argon2-cffi-bindings      21.2.0
arrow                     1.3.0
asttokens                 2.4.1
async-lru                 2.0.4
async-timeout             4.0.3
attrs                     23.2.0
Babel                     2.14.0
beautifulsoup4            4.12.3
bleach                    6.1.0
blis                      0.7.11
catalogue                 2.0.10
certifi                   2023.11.17
cffi                      1.16.0
charset-normalizer        3.3.2
click                     8.1.7
cloudpathlib              0.16.0
comm                      0.2.1
confection                0.1.4
contourpy                 1.2.0
cycler                    0.12.1
cymem                     2.0.8
datasets                  2.17.1
debugpy                   1.8.0
decorator                 5.1.1
defusedxml                0.7.1
dill                      0.3.8
evaluate                  0.4.1
exceptiongroup            1.2.0
executing                 2.0.1
fastjsonschema            2.19.1
filelock                  3.13.1
fonttools                 4.47.2
fqdn                      1.5.1
frozenlist                1.4.1
fsspec                    2023.10.0
huggingface-hub           0.20.2
idna                      3.6
ipykernel                 6.29.0
ipython                   8.20.0
ipywidgets                8.1.1
isoduration               20.11.0
jedi                      0.19.1
Jinja2                    3.1.3
joblib                    1.3.2
json5                     0.9.14
jsonpointer               2.4
jsonschema                4.21.1
jsonschema-specifications 2023.12.1
jupyter                   1.0.0
jupyter_client            8.6.0
jupyter-console           6.6.3
jupyter_core              5.7.1
jupyter-events            0.9.0
jupyter-lsp               2.2.2
jupyter_server            2.12.5
jupyter_server_terminals  0.5.1
jupyterlab                4.0.11
jupyterlab_pygments       0.3.0
jupyterlab_server         2.25.2
jupyterlab-widgets        3.0.9
kiwisolver                1.4.5
langcodes                 3.3.0
MarkupSafe                2.1.4
matplotlib                3.8.2
matplotlib-inline         0.1.6
mistune                   3.0.2
mpmath                    1.3.0
multidict                 6.0.5
multiprocess              0.70.16
murmurhash                1.0.10
nbclient                  0.9.0
nbconvert                 7.14.2
nbformat                  5.9.2
nest-asyncio              1.5.9
networkx                  3.2.1
nltk                      3.8.1
notebook                  7.0.7
notebook_shim             0.2.3
numpy                     1.26.3
overrides                 7.4.0
packaging                 23.2
pandas                    2.2.1
pandocfilters             1.5.1
parso                     0.8.3
pexpect                   4.9.0
pillow                    10.2.0
pip                       23.3.1
platformdirs              4.1.0
preshed                   3.0.9
prometheus-client         0.19.0
prompt-toolkit            3.0.43
psutil                    5.9.8
ptyprocess                0.7.0
pure-eval                 0.2.2
pyarrow                   15.0.0
pyarrow-hotfix            0.6
pycparser                 2.21
pydantic                  2.5.3
pydantic_core             2.14.6
Pygments                  2.17.2
pyparsing                 3.1.1
python-dateutil           2.8.2
python-json-logger        2.0.7
pytz                      2024.1
PyYAML                    6.0.1
pyzmq                     25.1.2
qtconsole                 5.5.1
QtPy                      2.4.1
referencing               0.32.1
regex                     2023.12.25
requests                  2.31.0
responses                 0.18.0
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rpds-py                   0.17.1
safetensors               0.4.1
scikit-learn              1.4.1.post1
scipy                     1.13.0
Send2Trash                1.8.2
setuptools                68.2.2
six                       1.16.0
smart-open                6.4.0
sniffio                   1.3.0
soupsieve                 2.5
spacy                     3.7.2
spacy-legacy              3.0.12
spacy-loggers             1.0.5
srsly                     2.4.8
stack-data                0.6.3
sympy                     1.12
terminado                 0.18.0
thinc                     8.2.2
threadpoolctl             3.4.0
tinycss2                  1.2.1
tokenizers                0.15.0
tomli                     2.0.1
torch                     2.1.2
tornado                   6.4
tqdm                      4.66.1
traitlets                 5.14.1
transformers              4.36.2
typer                     0.9.0
types-python-dateutil     2.8.19.20240106
typing_extensions         4.9.0
tzdata                    2024.1
uri-template              1.3.0
urllib3                   2.1.0
wasabi                    1.1.2
wcwidth                   0.2.13
weasel                    0.3.4
webcolors                 1.13
webencodings              0.5.1
websocket-client          1.7.0
wheel                     0.41.2
widgetsnbextension        4.0.9
xxhash                    3.4.1
yarl                      1.9.4

The dataset used is found [here](https://drive.google.com/drive/folders/10uQI2BZrtzaUejtdqNU9Sp1h0H9zhLUE). Please set up your file directory the following way for the code to work: Create a "Project" folder in your repository, then create a "QQA_Data" folder within this "Project" folder. Place the dataset files within the "QQA_Data" folder. Your notebook should be outside of the parent "Project" folder for the datasets to be imported correctly. 

Once complete, the directory should take on the following form:
base:
- Project3Code.ipynb
- Project:
- - QQA_Data: 
  - - QQA_dev.json
    - QQA_test.json
    - QQA_train.json

Where "base" is the directory containing the repository's files. 

We then recommend installing some base and NLP packages (most recent versions) using ```pip install jupyter torch numpy matplotlib``` and ```pip install nltk spacy transformers``` We also recommend users running into issues install the latest datasets, transformers (At least version 4.11.0), and scikit-learn using ```pip install datasets transformers``` and ```pip install -U scikit-learn```. The first few blocks in the notebook are setting up for the rest of our code base. We start with a few imports and then loading our datasets. 

We then manipulate our datasets to fit the tokenization methods we are using later on. The manipulation of our dataset occurs from the notebook block 3 until block 7. Here we are doing the following to preprocess our dataset: remove variant questions and changing the answer column to either a 1 or 0 (1 for Option 2, 0 for Option 1). This leaves us with a dataset that has 4 features: A question, choice 1, choice 2, and a label that is our correct answer. We then tokenize our data so we can set up for training. This set up ensures that new users do not have to adjust much if anything at all to preprocess our datasets. Just click run and go.

#### Part 2: Setting Up for Training

While running the blocks sequentially, we arrive to the sections with ```AutoModelForMultipleChoice``` and ```DataCollatorForMultipleChoice```. Both of these sections set up our models to fit with our adjusted datasets by setting them up in a multiple choice fashion. Next we set up our function to compute our evaluation metrics (In this case we are using F1 sccore to evaluate our models). 

#### Part 3: Training

The next section we come to is where we start to train our model. We start with setting up our model and providing arguments, encoded datasets from our training and evaluation, our tokenizer, our data collector, and finally we compute the evaluation metrics. Next we have a test block to ensure our trainer is working correctly, then we move on to our evaluator and formatting function. We then append each reference and prediction and evaluate how accurate our model is and finally print this number in a human readable format. We next define our ```trainAndEval``` function to combine our previous set up training with our evaluator

The system will train a model for 3 epochs on the train split, using the validation split to validate data. Each epoch, it will save a checkpoint to the same directory as the notebook, with a unique directory for each model--each model's directory is based on its model name. It will print its results to the cell as an output. 

#### Part 4: Evaluating the Models

The next section of code blocks in sequential execution are setting up and evaluating different out-of-the-box models from Huggingface. In order here are the models fine-tuned and evaluated:

###### Models
- bert-case-uncased
- distilbert
- bert-base-spanish-wwm-cased
- Roberta-base-squad2
- dynamic-tinybert
- distilbert-base-uncased-distilled-squad
- distilbert-base-cased-distilled-squad
- twitter-roberta-base-sentiment-latest
- feel-it-italian-sentiment
- Finance-Sentiment-Classification
- reviews-sentiment-analysis

We print the results from the evalutions after 3 epochs of training. This may take considerable compute time. 

#### Part 5: Results and Analysis

The outputs are the answer choice (1 or 0 from our earlier preprocessing) that are then evaluated for accuracy using F1 score. This is hardcoded so please do not forget to change these when you run your versions. These evaluation results are then stored in lists to more easily graph the results. The next blocks of code are bar charts of our grouped model types of Baseline, Sentiment Analysis, and SQUaD. we then graph the best from these categories in the last bar chart. See [here](https://docs.google.com/presentation/d/1K4x0OJyhAfyJciX1ozsdWsNbCaIEuzyqk6jUJZNDq2g/edit?usp=sharing) for our in class materials discussing our procedure and results.

### Final Results Expected Performance

Here are the expected test set performances for our models (accuracies) after fine-tuning and evaluation:
- bert-base-uncased: 0.5123456790123457
- distilbert: 0.4876543209876543
- bert-base-spanish-wwm-cased: 0.5370370370370371
- Roberta-base-squad2: 0.5246913580246914
- dynamic-tinybert: 0.5308641975308642
- distilbert-base-uncased-distilled-squad: 0.49382716049382713
- distilbert-base-cased-distilled-squad: 0.5061728395061729
- twitter-roberta-base-sentiment-latest: 0.5185185185185185
- feel-it-italian-sentiment: 0.49382716049382713
- Finance-Sentiment-Classification: 0.5246913580246914
- reviews-sentiment-analysis: 0.5617283950617284
