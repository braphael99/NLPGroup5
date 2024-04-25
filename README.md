# NLPGroup5

## Project Contributors:

- Anand Odbayar: Contributor (Technical Reports, Final Presentation)
- Blake Raphael: Core Contributor (Dataset Acquisition, Evaluating, Results Gathering, Technical Reports, Final Presentation)
- Giovanni Evans: Core Contributor (Dataset Acquisition, Evaluating, Results Gathering, Technical Reports, Final Presentation)
- Patrick Liu: Lead (Dataset Acquisition and Maniupulation, Training, Evaluating, Results Gathering, Technical Reports, Final Presentaion)
- Reilly Cepuritis: Core Contributor (Results Gathering, Technical Reports, Evaluating, Final Presentation)

### Further Breakdown:

- Anand Odbayar: Anand helped contribute towards the final presentation. He created the Procedure and Analysis slides. 
- Blake Raphael: Blake contributed by finding three of our models that we tested, specifically Dynamic TinyBERT, DistilBERT Base Cased SQUaD, and DistilBERT Base Uncased SQUaD. He also wrote up a bulk of our progress report including elaborating a bit on our methodology, finding three of our related works, and explaining our next steps in the process. He also created the Premise and Hypothesis Slides as well as helping with the Related Works slides and proofreading. Blake also helped proofread the project abstract along with finding the premise and dataset.
- Giovanni Evans: Giovanni Evans contributed towards the project by finding three of the models that were originally used in our project, but dropped due to poor results including RemBERT, Longformer, and MEGA. He also contributed to the analysis, comparison, and conclusion sections in the code implementation of the project. He also contributed to the project abstract, progress report, and final presentation. 
- Patrick Liu: Patrick contributed by implementing most of the code base and debugging. Patrick also helped by finding the premise and dataset along with writing up the project abstract. He found the bulk of the models to test and helped format results both in the code and on the slides. Patrick found the following models to test:  BERT base-uncased, RoBERTa base-sentiment, DistilBERT base-uncased, BERT base-spanish, RoBERTA base-SQUAD2, RoBERTa large-english sentiment, and Feel-it italian-sentiment. Patrick also created the Results section of our slides and contributed to the progress report by introducing our concept, discussing our dataset, outlining a bit of our methodology, finding a related work, and explaining a few of our next steps.
- Reilly Cepuritis: Reilly contributed by creating some of the analytical portions of the report. Reilly added some clarity working on the graphs, as well as spearheading the concluding elements of the report. Reilly also helped with organizing some of the group meetings. For the presentation, Reilly took a role in finalizing the conclusion, related works slide, as well as adding the updated graphs and proofreading for clarity. 

##### Terminology:

The roles are defined as below:

- Lead: Individual(s) responsible for the entire workstream throughout the project.
- Core Contributor: Individual that had significant contributions to the workstream throughout the project.
- Contributor: Individual that had contributions to the project and was involved with the effort.

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

All code was run in a Jupyter Notebook on an Anaconda enviroment. To install all requirements, navigate to the project directory and run "pip install -r requirements.txt".

The dataset used is found [here](https://drive.google.com/drive/folders/10uQI2BZrtzaUejtdqNU9Sp1h0H9zhLUE). Please set up your file directory the following way for the code to work: Create a "Project" folder in your repository, then create a "QQA_Data" folder within this "Project" folder. Place the dataset files within the "QQA_Data" folder. Your notebook should be outside of the parent "Project" folder for the datasets to be imported correctly. 

Once complete, the directory should take on the following form:
base:
- requirements.txt
- Project3Code.ipynb
- Project:
- - QQA_Data: 
  - - QQA_dev.json
    - QQA_test.json
    - QQA_train.json

Where "NLPGroup5-main" is the directory containing the repository's files. 

The initial cells load our dataset into three splits, using the QQA_train.json file as the "train" split, the QQA_dev.json file as the "validation" split, and the QQA_test.json file as the "test" split. 

The next cells manipulate our datasets to fit the tokenization methods we are using later on. The manipulation of our dataset occurs from the notebook block 3 until block 7. Here we are doing the following to preprocess our dataset: remove variant questions and changing the answer column to either a 1 or 0 (1 for Option 2, 0 for Option 1). This leaves us with a dataset that has 4 features: A question, Choice 1, Choice 2, and a label that is our correct answer. 

A preprocessing function is defined that transforms the question into two candidate sentences--each of which starts with the question and then ends with one of the two answers--before tokenizing them. This set up ensures that new users do not have to adjust much if anything at all to preprocess our datasets. Just click run and go.

#### Part 2: Setting Up for Training

While running the blocks sequentially, we arrive to the sections with ```AutoModelForMultipleChoice``` and ```DataCollatorForMultipleChoice```. Both of these sections set up our models to fit with our adjusted datasets by setting them up in a multiple choice fashion. Next we set up our function to compute our evaluation metrics (In this case we are using F1 score to evaluate our models). 

#### Part 3: Training

The next section we come to is where we start to train our model. We start with setting up our model and providing arguments, encoded datasets from our training and evaluation, our tokenizer, our data collector, and finally we compute the evaluation metrics. Next we have a test block to ensure our trainer is working correctly, then we move on to our evaluator and formatting function. We then append each reference and prediction and evaluate how accurate our model is and finally print this number in a human readable format. We next define our ```trainAndEval``` function to combine our previous set up training with our evaluator. 

The "trainAndEval" function takes a function name and batch size as input, and prints the evaluated F1 micro score once it finishes fine-tuning and evaluating. Prior to training, the system preprocesses our dataset using the pre-defined preprocessing function, and then trains a model for 3 epochs on the train split, using the validation split to validate data. Each epoch, it will save a checkpoint to the same directory as the notebook, with a unique directory for each model--each model's directory is based on its model name. Once training and evaluation are complete, the cell will print its results. 

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

We print the results from the evalutions to the cell's output after 3 epochs of training. This may take considerable compute time. 

#### Part 5: Results and Analysis

The printed evaluation results are then manually stored in lists to more easily graph the results: To save processing time and to avoid re-running cells, these are hardcoded and must be changed manually if evaluation results changed. The next blocks of code are bar charts of our grouped model types of Baseline, Sentiment Analysis, and SQuAD. we then graph the best from these categories in the last bar chart. See [here](https://docs.google.com/presentation/d/1K4x0OJyhAfyJciX1ozsdWsNbCaIEuzyqk6jUJZNDq2g/edit?usp=sharing) for our in class materials discussing our procedure and results.

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
