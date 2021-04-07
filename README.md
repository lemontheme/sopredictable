# sopredictable

*sopredictable* ('so predictable') lets you save the results of machine learning training 
runs  in such a way that they can be loaded
easily in a prediction setting.


## Getting Started

### Requirements

Python 3.7 or later. All OSs and architectures are supported.

### Installation

```
pip install sopredictable
```

## Basic usage





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
    |-- conda.yml   # optional
    |-- dockerfile  # optional
    |-- {code_path}.zip  # optional
    |-- artifacts/
        |-- {user-specified artifact name}/  # e.g. 'sklearn_pipeline'
            |-- cls.pkl  # pickled output of type(model_a)
            |-- data/  # the path provided to `{save,load}_artifact()` functions.
                |-- artifact.pkl  # actual state of model, e.g. parameters
        |-- ...
```

### Adding custom artifact persistence logic


```
pip install typicalpredictable
pip install sopredictable
pip install utterlyexpected
pip install easypretzel
pip install usablerelic
```
