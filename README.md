# Easy Pretzel

## Goals

I want to be able to save training run artifacts in such a way that they can be loaded
easily in a prediction setting.

```
# Pretzel archive (implicit)
# .
# |-- {pretzel_archive_name}
#     |-- meta.json
#     |-- conda.yml
#     |-- {code_path}.zip
#     |-- artifacts
#         |-- model_a/
#             |-- cls.pkl
#             |-- data/
#                 |-- whatever.pkl
```