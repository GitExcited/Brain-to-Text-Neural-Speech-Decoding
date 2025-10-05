proposal checklist:
 - Problem Statement and Application
 - What reading material
 - Possible Methodology
 - Metric Evaluation
 - Gantt Chart
 - Bibliography
 - Formatting


```
In Kaggle challenges https://www.kaggle.com/competitions, you
are expected to choose an open competition, which requires you to study the problem
and propose a solution. Note that you must choose an open competition with image or
text or speech data to apply deep learning models there. You should also check the data
size because the project should not require a lot of computational power i.e. GPU. Note
that in this track, you will be judged on your ability to apply a methodology and a sound
solution that works well in a challenging problem. 
```


You should write a one-page proposal for the course project to cover the following topics:

## Problem Statement and Application: (Ion Turcan 40154098)

```
provide a background about the topic to be
investigated and specify why the problem is interesting and important? What are the
associated challenges of the problem application? What are your expectations/goals
throughout developing the application of interest?
```



todo: re-structure into parts

In this project we are going to investigate the topic of linguistic neural decoding. The problem is interesting and important because many people that had a brain stroke or suffer from ALS could lose the capability of talking. To restore speech, the patient would need an interface that would capture the brain activity related to speech. Our main goal is to create a model that would be able to decode this brain activity data. For that we would use deep learning architecture such as RNN. For this part we expect to end up with a model that would have an accuracy of 70 % at least. This is a challenge because that involves setting up a working pipeline. After accomplishing this task, our secondary goal would be to try to improve this base model through different improvement strategies. These strategies might include data augmentation, model architecture change, using different loss function and using different tokenization strategies. The results are going to be reported and discussed. This secondary task is really challenging because that involves deeper understanding of the whole pipeline and it is very likely that we will need to consult additional materials to have an idea on how these potential improvements could be implemented.
    
## What reading material (Kirill)

Draft:<br />
[https://ojs.aaai.org/index.php/AAAI/article/view/5146](https://arxiv.org/abs/2501.04844?)
The model in this paper is very similar to the kaggle's baseline model. It uses a speech module which is not applicable for our project though.
This paper serves as a good overview of the current SOA for EEG/ speech decoding.
- EEG Module: (This trains the model to understand EEG structure)
  - EEG Encoder: Compresses raw EEG signals into "latent representations"
  - EEG Decoder: Reconstructs EEG latent representation into raw EEG, acts as a self-supervised learning constraint.
- Phoneme Predictor: Outputs phoneme probabilites by trying to predict phoneme sequences (/p/, /a/, /t/) from compressed EEG signals.
{training phonemes: EEG (signals) <-> Encoder (latent)} -> Phoneme Predictor -> Phoneme probabilities.

[https://theaisummer.com/speech-recognition/](https://theaisummer.com/speech-recognition/)
Not really a research paper, but this article reviews/explains various DL architectures for speech recognition. 
It explains and justifies the use of RNN's and CTC-based models (which is what the kaggle baseline uses), and compares the performance of different models while providing suggestions while citing other papers.<br />

[https://arxiv.org/abs/2112.09239](https://arxiv.org/abs/2112.09239)<br />
This paper explores how a Transformer-based architectures can outperform RNNs for EEG sequence decoding which could be explored as an alternative to GRUs for modeling relationships between data points across time in precomputed EEG features, like those in the Kaggle dataset.

[https://pmc.ncbi.nlm.nih.gov/articles/PMC11030484/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11030484/)<br />
This paper is authored by the same team hosting the Kaggle competition. It largely forms the basis of the competition's baseline model architecture.

the authors of the challenge made a paper we should check
they also made a previous paper for the previous challenge
we can check what other stuff they have written
we can check what papers their papers refer to and add them to our reading list


- read the abstracts
- keywords:
  - brain signals

- research process
  - gather papers
  - learn domain
  - learn existing methods attempted
  - find good ones
  - find general methodology used in papers
  - find baselines used in papers
  - benchmark datasets
  - shared datasets
  - share repo's
  - make a list of potential models / methods to attempt
    - maybe simpler or less explored



## Possible Methodology: 
```
highlight the possible method or algorithm you are proposing.
Are there any existing implementations to be used and how will you use them? How are you
planning to improve or modify such implementations? You may not have the exact answer
here but try to give an answer that you will follow as much as possible.
```

asdf

- technical process
  - run existing benchmarks: load without training and infer
    - Stanford-NPTL causal RNN Ensemble + 5gram
    - Stanford-NPTL causal RNN TTA-Ensemble + 5gram
    - UCD-NPL causal RNN + 5gram
    - todo: write a submission script / module (prolly does not exist)

  - build (-CHOOSE- / FIND) baselines
    - naive / simple models / fast to train
     - random
     - mean
     - median
     - linear regression
     - logistic regression
     - knn
     - svm
  - gather the list of suggested improvments
    - assess the list of improvments
    - find from the kaggle page
      -  authors have suggested stuff
  - create our own list of potentially better models / architectures
    - from research papers 
  - create our own list of improvments on the existing models
    - hyperparam tuning methods
    - data preprocessing
    - data supplementing
    - data augmenting
    - loss functions
- clone the benchmarks into our repo for accees and version control
  - track and modify the benchmarks
  - swap out parts
    - optimiser
    - ...

- familiarize with kaggle and add entries to the leaderboard
  - -submit titanic dataset-
  - submit dummy csv (done)
  - submit benchmark results (loaded, not trained)
  - train and submit benchmark models
  - gather links to papers onto this repository
    - (optionnal) use zotero


todo:
- clone research repo
- set up codespaces / colab / kaggle env


## Metric Evaluation ( david ): 
```
Discuss how you will evaluate your results both in terms of
qualitative and quantitative analysis. Qualitatively, what kind of results do you expect (e.g.
plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or
compare your results with (e.g. what performance metrics or statistical tests)? All these
metric evaluations will be used to assess and evaluate the pipeline and your expectations
regarding the kind of results/performance to be achieved.
```
We expect a .csv file of our model predictions for the dataset  of 1,450 test sentences for this challenge. Our output is therefore sentence predictions based on the input of neural recordings. We will use the performance metric `word error rate` which is defined in the challenge as the "Edit distance between the decoded sentence and the actual sentence, computed over words". This basically means the number of edits such as substitutions, insterions and deletions necessary to make the predicted sentence match the true sentence. We will also be using loss as an ongoing metric during training to determine the confidence of our model in its predictions and to evaluate how loss decreases as the training batches increases.

## Gantt Chart (David) (to be discussed, add your suggestions and we conclude tomorow): 

```
use an additional page (supplemental material) to illustrate a Gantt
chart of the project development to list (a) schedules and (b) items of milestones and
deliverables. Note that you cannot use this page to extend your proposal description.
```
## 🗓️ Gantt Chart — EEG-to-Speech Kaggle Project

| **Phase / Task** | **Description** | **Responsible** | **W1** | **W2** | **W3** | **W4** | **W5** | **W6** | **W7** | **W8** | **Deliverable / Milestone** |
|------------------|-----------------|-----------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|------------------------------|
| Project Setup | Choose competition, define scope, assign roles | All | ✅ |  |  |  |  |  |  |  | Selected Kaggle competition |
| Proposal Draft | Write and format proposal (Problem, Reading, Methodology, Metrics, Bibliography) | All (Lead: Ion) | ✅ | ✅ |  |  |  |  |  |  | Submission-ready proposal |
| Literature Review | Collect and summarize key papers (arXiv, PMC, Kaggle baselines) | Kirill |  | ✅ | ✅ | ✅ |  |  |  |  | Curated reading list |
| Environment Setup | Configure Kaggle notebooks, Colab / Codespaces; clone baselines | David |  | ✅ | ✅ |  |  |  |  |  | Reproducible environment setup |
| Benchmark Exploration | Run and analyze provided Kaggle benchmark (baseline RNN models) | All |  |  | ✅ | ✅ |  |  |  |  | Initial benchmark submission |
| Baseline Validation | Train and validate baseline models; verify WER metric | Elion & Ion |  |  |  | ✅ | ✅ |  |  |  | Working baseline & leaderboard score |
| Model Improvement | Explore architecture changes (Transformers, GRU variants), hyperparameter tuning | All |  |  |  |  | ✅ | ✅ | ✅ |  | Improved model & documentation |
| Evaluation & Visualization | Analyze training logs, WER trends, qualitative outputs | Kirill & David |  |  |  |  |  | ✅ | ✅ |  | Evaluation report & figures |
| Final Report & Presentation | Compile final paper (LaTeX/Sphinx) & presentation slides | Elion & Ion |  |  |  |  |  |  | ✅ | ✅ | Final report & submission |


## Bibliography (ELION) : 
```
use an additional page to extend your reference list cited in your
proposal. The citations may include, but not limited to, published papers and domain links.
(include a link to your dataset). Please note that failure to properly cite your references
constitutes to a plagiarism.
You will be given the opportunity to submit your proposal for revision by the professor/LeadTA, before the final graded submission. Only the admin (one person) of your team needs to
upload the proposal in PDF file in Moodle.
```
todo: 
- compile list of papers into bibtex / latex


## Formatting (LATEX / SPHINX) [ELION] :
```
For the report format, please consult “Reports Formatting” Section in this guideline. Our
team (TAs and lecturer) will review your proposal and, if it is acceptable, you may proceed
with developing the next phase of your project. Otherwise, we will instruct you to either revise
or re-write the proposal according to the guidelines of the course project. All teams are highly
encouraged to put great eƯort on preparing the first proposal draft to avoid further delays in
project developments. 
```
