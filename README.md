Work in progress. Don't use. 

# sopredictable [WIP]

*sopredictable* ('so predictable') lets you save the results of machine learning training 
runs  in such a way that they can be loaded
easily in a prediction setting.

## Getting Started

### Requirements

- Python 3.7 or later.
- All OSs and architectures are supported.

### Installation

```
# pip install sopredictable
```

## Basic usage

TODO



## Concepts

### The *Predictable* Interface

Python protocols. No subclassing. Bring your own implementation. Design by contract.

A typical predictor. Second mode of machine learning model, after training. Compare with
abstractions in AllenNLP, GCP AI platform, cortex, bentoml and mlflow.

### Representation of a Predictor on Disk

```
.
|-- {user-specified directory}/
    |-- meta.json
    |-- conda.yml   # optional. env setup is user's responsibility
    |-- dockerfile  # optional. env setup is user's responsibility
    |-- {code_path}.zip  # optional (appended to PYTHONPATH when loaded)
    |-- artifacts/
        |-- {user-specified artifact name}/  # e.g. 'sklearn_pipeline'
            |-- cls.pkl  # pickled output of `type(model_a)`
            |-- saveload.pkl # (optional) pickled tuple `(save_fn, load_fn)`
            |-- data/  # the path provided to `{save,load}_artifact()` functions.
                |-- artifact.pkl  # actual state of model, e.g. parameters
        |-- ...
```

Inside meta.json:

```
{
    "metrics": {"accuracy": 0.89},
    "hparams": {"lr": 0.02, "initial_dropout": 0.5},
    "dataset": {"source": "disaster-tweets"},
    "tags": ["multilabel", "cnn"],
    "_created_at": "2021-02-21T23:21:02.412Z",
    "_sopredictable_version": "0.1.0",
    "_python_version": "3.9.2",
    "_os": "macOS",
    "_artifacts": [
        "model": {"cls": ..., "save": ..., "load": ..., "deps": ...}
    ]
}
```

or (better):

```json
{
    "_model": {
        "metrics": {"accuracy": 0.89},
        "hparams": {"lr": 0.02, "initial_dropout": 0.5},
        "dataset": {"source": "disaster-tweets"},
        "tags": ["multilabel", "cnn"],
    },
    "_created_at": "2021-02-21T23:21:02.412Z",
    "_sopredictable_version": "0.1.0",
    "_python_version": "3.9.2",
    "_os": "macOS",
    "_artifacts": [
        "model": {"cls": ..., "save": ..., "load": ..., "deps": ...}
    ]
}
```



### Adding custom artifact persistence logic


```
pip install typicalpredictable
pip install sopredictable
pip install utterlyexpected
pip install easypretzel
pip install usablerelic
```


## License

Distributed under the MIT License. See LICENSE for more information.

## Acknowledgements

- BentoML
- MLFlow
- AllenNLP
- GCP AI Platform
- Cortex


