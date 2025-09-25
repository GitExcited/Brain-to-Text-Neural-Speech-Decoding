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
  - [Member 2] 
  - [Member 3]  

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
| Proposal  | Sept 15 | Initial repo setup + project plan |
| Milestone 1 | Oct 15 | Working data loader + baseline model |
| Milestone 2 | Nov 15 | Improved model + validation pipeline |
| Final Submission | Dec 31 | Complete repo + report + Kaggle submission |

##  Credits
Organized by UC Davis Neuroprosthetics Lab.  
Competition link: [Kaggle Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25)
