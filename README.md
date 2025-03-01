# Spark Project: Understanding Emotions in IMDB Movie Reviews

## Overview

**Goal:**
The goal of this project is to understand the relationship between **emotions**, **ratings**, and **keyword frequency** in movie reviews. By analyzing these patterns, we aim to enable **personalized and meaningful recommendations** based on user feedback.

This project uses **Apache Spark** to analyze the IMDB reviews dataset from [Kaggle](https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset) (7.78 GB). The project consists of two main analyses:

1. **Emotion Analysis**

    - Leveraged the Hugging Face model [ALBERT Base V2 Emotion](https://huggingface.co/bhadresh-savani/albert-base-v2-emotion) to classify emotions in movie reviews.
    - Identified emotional patterns across reviews to correlate them with user ratings and keywords.

2. **N-gram Analysis**
    - Used Spark's ML library (`org.apache.spark.ml.feature.NGram`) to analyze **keyword frequency** in the reviews.
    - Extracted meaningful n-grams to study their relationship with emotions and ratings.

---

## Execution Environment

-   **Spark Cluster**: The analyses were executed on **Google Dataproc**, as part of the NYU curriculum.
-   **Resources**: The code was run on **CPUs**, which introduced **resource constraints**, leading to incomplete processing of the dataset.
    -   To process more data, the **partition size** can be increased (configurable in `process_emotion.py` as `--partitions`).
    -   For optimal performance, it is recommended to run the project on **GPU-enabled clusters**, as the ALBERT model is computationally intensive and better suited for GPUs.

---

## Folder Structure

-   **`results/`**: contains the analysis results as CSV files.

-   **`zeppelin-notebooks/`**: for ngram analysis and generating the analysis csv data

-   **`process_emotion_cluster_mode.sh`** for running the emotion processing in cluster mode.

## Environment Setup

### Create the virtual environment

```
python -m venv .venv
source .venv/bin/activate

pip install pyspark torch transformers huggingface_hub venv-pack
```

Package the environment into a tar.gz archive:

```
(.venv) $ venv-pack -o environment.tar.gz
Collecting packages...
Packing environment at
```

### Start Spark Shell

To launch the interactive job, do the following within the virtual environment,

```
PYSPARK_DRIVER_PYTHON=`which python` \
PYSPARK_PYTHON=./environment/bin/python \
pyspark \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
--master yarn \
--deploy-mode client \
--archives environment.tar.gz#environment

```

You can use the below to check your environment

```
import sys
print(sys.executable)

```

### Check spark application

Check running application

```
yarn application -list
```

Kill running application, for example

```
yarn application -kill application_1724767128407_12034
```
