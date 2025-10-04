#  Brain-to-Text '25: Neural Speech Decoding

Decode intracortical neural activity during attempted speech into words.  
This repo contains our work for the COMP 432 Project. Our chosen Kaggle comp is the [Kaggle Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25) challenge.

---

## Project Overview
Speech brain-computer interfaces (BCIs) aim to restore communication for people with ALS or brainstem stroke by decoding speech directly from brain activity.  
Our goal: **develop deep learning models to map neural spiking activity → text**.

## 👥 Team Information
- **Course:** COMP 433 – Fall 2025  
- **Members:**
  - J. David Ruiz (40176885)
  - Elion Abdyli (40132982) 
  - Ion Turcan (40154098)
  - Kirill Vishnyakov (40281175)

> Detailed contributions are also included in the final project ZIP, as per course requirements.

##  Repo Structure
```
notebooks/       # Colab notebooks (baseline, experiments, submissions)
src/             # Core scripts (data loading, models, utils)
submissions/     # Kaggle-ready .csv outputs
requirements.txt # Dependencies
```


##  Getting Started
Clone the repo:
```bash
git clone https://github.com/GitExcited/Brain-to-Text-Neural-Speech-Decoding
cd brain-to-text-25
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Open a Colab notebook:
- Upload the desired `.ipynb` from `notebooks/`
- Connect to Kaggle dataset
- Run experiments / generate submissions


##  Competition
- **Challenge:** Map neural activity to text (no ground-truth alignment).
- **Metric:** Word Error Rate (WER).
- **Deadline:** December 31, 2025.

Baseline: ~6.7% WER  
Our goal: **beat the baseline.**

## 📅 Project Milestones (COMP 433)
| Milestone | Date | Deliverable |
|-----------|------|-------------|
| Proposal  | Oct 5 | Initial repo setup + project plan and research |
| Milestone 1 | Nov 2 | Progress Reporting |
| Milestone 2 | Nov 30 | Final Reporting |
| Final Presentation | Nov 30 |
| Github Submission | Nov 30 | 




##  Credits
Organized by UC Davis Neuroprosthetics Lab.  
Competition link: [Kaggle Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25)

# discussion
asdf
- research process
  - gather papers
  - learn domain
  - learn existing methods attempted
  - find good ones
  - find general methodology used in papers
  - find baselines used in papers
  - make a list of potential models / methods to attempt

- technical process
  - run existing benchmarks
    - Stanford-NPTL causal RNN Ensemble + 5gram
    - Stanford-NPTL causal RNN TTA-Ensemble + 5gram
    - UCD-NPL causal RNN + 5gram
  - build baselines
  - gather the list of suggested improvments
    - assess the list of improvments
  - create our own list of potentially better models / architectures
  - create our own list of improvments on the existing models
    - hyperparam tuning methods
    - data preprocessing
    - data supplementing
    - data augmenting
    - loss functions
- clone the benchmarks into our repo for accees and version control
  - track and modify the benchmarks
