# LibreLog: Accurate and Efficient Unsupervised Log Parsing Using Open-Source Large Language Models

This repo is the package for [ICSE2025] "LibreLog: Accurate and Efficient Unsupervised Log Parsing Using Open-Source Large Language Models"

## Overall workflow of LibreLog

<p align="center"><img src="docs/work_flow.png" width="1000"></p>

## Structure
We present LibreLog repository structure below.

```
.
├── README.md
├── docs
│   └── work_flow.pdf
├── evaluation
│   ├── RQ1
│   │   └── RQ1.png
│   ├── RQ2
│   │   └── RQ2.png
│   ├── RQ3
│   │   ├── RQ3_1.png
│   │   └── RQ3_2.pdf
│   └── RQ4
│       └── RQ4.png
├── full_dataset
│   └── README.md
├── models
│   └── README.md
├── parser
│   ├── accuracy.py
│   ├── evaluator.py
│   ├── evaluator_logbase.py
│   ├── grouping.py
│   ├── llama_parser.py
│   └── regex_manager.py
├── parsing.sh
├── parsing_logbase.sh
├── requirements.txt
└── results
    ├── AEL.csv
    ├── Drain.csv
    ├── LILAC.csv
    ├── LLMParsert5base.csv
    └── LibreLog.csv
```


## Requirement 

You need to have `uv` installed to manage the python environment.

```shell
uv sync
```

## Models download

Please download the base LLM (Meta-Llama-3-8B-Instruct) from [Huggingface](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Remember that you need a huggingface login and apply to access this gated model.

```
hf auth login
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir models/Meta-Llama-3-8B-Instruct
```

## Datasets download

This fork of LibreLog is able to run on Loghub-2.0 as well as LogBase.

Please first download the full datasets of Loghub-2.0 via [Zenodo](https://zenodo.org/record/8275861) and place it inside the `full_dataset` folder.
Download also [LogBase](https://doi.org/10.6084/m9.figshare.28815620) and unzip it into a `LogBase` folder.

## Parsing

Please run the following command to run LibreLog on Loghub-2.0.
```shell
sh parsing.sh
```

Please run the following command to run LibreLog on LogBase.
```shell
sh parsing_logbase.sh
```