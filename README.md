# Recommender Systems Challenge 2020/2021 - PoliMi

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/recommender-system-2020-challenge-polimi)

Part of the Recommender Systems exam at Politecnico di Milano consists in a kaggle challenge. In this repository you can find all the files that we used for the competition. 

Note that recommenders come from [this repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi).

## Overview
The complete description of the problem to be solved can be found in the kaggle competition link (check the top of the read.me). Shortly, given the User Rating Matrix and the Item Content Matrix we had to **recommend 10 potentially relevant books** to the users. In particular the URM was composed by around **135k interactions**, **7947 users** and **25975 item**; the ICM instead contained, for each book, a subset of **20000 possible tokens** (some books had less than 10 tokens, other more than 40).

Note that the evaluation metric for this competition was the mean average precision at position 10 (**MAP@10**).

## Our Best Model
Our final model used for the best submission is an hybrid model composed by different models with different hyperparameters for **different users profiles**.
In particular, we used **ItemCF, RP3Beta and ALS** models tuned on 3 different tiers of users (users with less than 8 interactions, users with more than 20 interactions and users in the middle).

<p align="center">
    <img src="https://i.imgur.com/Ptggytw.png" width="900" alt="Politecnico di Milano"/>
</p>

As presented above, the process to build the final model was incremental; the model that alone performed in the best way was the Item Collaborative Filtering. However, **merging the score** of that with the other models we could obtain better and better results. 

The following image represents in which way we merged and we normalized the scores of the models for the **3 tiers of users** (a green circle stands for normalization, a plus stands for a weighted merge of the scores).

<p align="center">
    <img src="https://i.imgur.com/jbBe1tF.png" width="350" alt="Politecnico di Milano"/>
</p>

## Recommenders
In this repo you can find the implementation of different recommender systems; in particular the following models can be found in the *Recommenders* folder:
- Item and User based Collaborative Filtering
- Item Content Based Filtering
- P3Alpha and RP3Beta Graph Based models
- Pure SVD and Implicit Alternating Least Squares models
- Slim BPR and Slim ElasticNet

## Requirements
The `requirements.txt` file lists all the libraries that are necessary to run the scripts. **Install them** using:

```
pip install -r requirements.txt
```

## Cython
Some of the models use Cython implementations. As written in the original repository you have to **compile all Cython algorithms**. 
In order to compile you must first have installed: gcc and python3 dev. Under Linux those can be installed with the following commands:

```
sudo apt install gcc 
sudo apt-get install python3-dev
```
If you are using Windows as operating system, the installation procedure is a bit more complex. 
You may refer to [the official guide](https://github.com/cython/cython/wiki/InstallingOnWindows).

Now you can compile all Cython algorithms by running the following command. 
The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. 
During the compilation **you may see some warnings**. 

```
python run_compile_all_cython.py
```

## Results
With our best submission we had the following results:
- Ranked **1st** on public leaderboard (score: **0.10103**)
- Ranked **3rd** on private leaderboard (score: **0.10796**)

## Team
- Samuele Meta [[Github](https://github.com/SamueleMeta)] [[Email](mailto:metasamuele@gmail.com)]
- Stiven Metaj [[Github](https://github.com/StivenMetaj)] [[Email](mailto:stivenmetaj@gmail.com)]
