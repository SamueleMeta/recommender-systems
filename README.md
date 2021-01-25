# Recommender Systems Challenge 2020-2021 - PoliMi

[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/recommender-system-2020-challenge-polimi)


Part of the Recommender Systems exam at Politecnico di Milano consists in a kaggle challenge. In this repository you can find all the files that we used for the competition. 

Note that recommenders come from [this repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi).

---

## Requirements
The `requirements.txt` file lists all the libraries that are necessary to run the scripts. Install them using:

```
pip install -r requirements.txt
```

## Cython
Some of the models use Cython implementations. As written in the original repository you have to compile all Cython algorithms. 
In order to compile you must first have installed: gcc and python3 dev. Under Linux those can be installed with the following commands:

```
sudo apt install gcc 
sudo apt-get install python3-dev
```
If you are using Windows as operating system, the installation procedure is a bit more complex. 
You may refer to [the official guide](https://github.com/cython/cython/wiki/InstallingOnWindows).

Now you can compile all Cython algorithms by running the following command. 
The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. 
During the compilation you may see some warnings. 

```
python run_compile_all_cython.py
```
