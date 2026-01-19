This is the replication package for **MCRead** model titled *"MCRead: A Multi-Granularity Context-Aware Approach for Code Change Readability Assessment"* including both code and datasets. 


# Project Structure
```
MCRead 
├─core
│  ├─comp 
│  │  ├─dataset_readers
│  │  │      readers.py     
│  │  ├─metric        
│  │  ├─nn           
│  │  └─tokenizers       
│  └─models       
├─evaluation           
├─models
├─tasks
│  │  aggre_results.py
│  │  evaluate.py
│  │  train_helper.py
│  └─configs
│      base.jsonnet         
└─utils
```
In `dataset_readers` folder, `readers.py` implements data preprocessing. The files in the `tasks` folder are used for training, testing, and evaluating the model, while the `configs/base.json` file serves as the configuration file for the model.

# Requirements

## Python Package Dependencies
You should al least install following packages to run our code:
- PyTorch: 1.8.0+cu111
- allennlp: 2.8.0
- allennlp_models: 2.8.0
- nltk: 3.5
- ...

The full list of dependencies is listed in `requirements.txt`.


# Run MCRead
## Data Preparation
Due to size limit, we archive our data in the Google Drive and you can download the data from this link: [data.zip](https://drive.google.com/file/d/19_YBvU6J2tBnxr47NOpUt0NWLnoLefq9/view?usp=drive_link). 
Unzip this file and move the `data` folder to the root of this project to finish preparation of data.

## Before Running
To ensure some scripts will work well, you have to do three  things first:
1. Open "base_global.py" and check the path of Python interpreter, you should configure the right python interpreter path here.
2. **Make sure you are running all the code at the root directory of the CCRep project**, this is important. 
3.You need to download the token-level pre-trained model **codebert_base** and the character-level pre-trained model **canine-s**, and place them in the root directory of the current project using the pre-trained model names as folder names.


## Running
Execute follow command **at the root of the project** to run the apca task:
```shell
python tasks/train_helper.py 
```


The script will automatically do training, testing and report the final performance.

