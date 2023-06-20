# Unified Reverse Dictionary and Definition Modelling 

### Overview
This repository contains the code and human evaluation outputs for our research on a multi-task model to unify reverse dictionary and definition modelling. The research has been publiched as:
- [A unified model for reverse dictionary and definition modelling](https://aclanthology.org/2022.aacl-short.2/) at AACL-IJCNLP 22
  - model details, training objectives, experiments on two relatively high-resource English datasets, human evaluation.
- [Edinburgh at SemEval-2022 Task 1: Jointly fishing for word embeddings and definitions](https://aclanthology.org/2022.semeval-1.8/) at SemEval 2022
  - experiments and analysis on five languages and three types of embeddings, in a low-resource scenario; discussions on the tasks.
  - it was a winning submission and won best paper honourable mention out of 221 papers.

### Data
The two papers above used data from different sources. Unfortunately, we do not own any of the data, so please refer to the original papers or links below.
- reverse dictionary and deifinition modelling in 5 languages: [CODWOE 2022@SemEval 2022](https://aclanthology.org/2022.semeval-1.1/)
- reverse dictionary in English: [HILL](https://github.com/thunlp/MultiRD)
- definition modelling in English: [CHANG](https://aclanthology.org/D19-1627/)

### Run

```
    conda env create -f environment.yml # creates a conda environment named 
"unified" which has the packages needed to run the code
```

```
    # unified model with shared word embeddings and output layer.
    python code/train_unfied_share.py \
      --batch_size ${BSZ} \
      --save_dir ${SAVE_MODEL_DIR} \
      --summary_logdir ${SAVE_LOG_DIR} \
      --do_train \
      --train_file ${TRAIN_FILE} \
      --dev_file ${DEV_FILE} \
      --embedding_arch ${ARCH} \ # (sgns, char, electra, bert)
      --device ${DEVICE} \
      --do_pred \
      --test_file ${TEST_FILE} \
      --pred_file ${SAVE_PRED_DIR} \
      --pred_direction ${DIRECTION} \ # (embedding, definition)
      --skip
```

### References

```
@inproceedings{chen-zhao-2022-unified,
    title = "A Unified Model for Reverse Dictionary and Definition Modelling",
    author = "Chen, Pinzhen  and Zhao, Zheng",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-short.2",
    pages = "8--13",
}

@inproceedings{chen-zhao-2022-edinburgh,
    title = "{E}dinburgh at {S}em{E}val-2022 Task 1: Jointly Fishing for Word Embeddings and Definitions",
    author = "Chen, Pinzhen  and Zhao, Zheng",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.8",
    doi = "10.18653/v1/2022.semeval-1.8",
    pages = "75--81",
}
```
