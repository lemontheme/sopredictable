from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Sequence

from sopredictable import Saver, load, alias


class Model:
    def __call__(self, x: int) -> int:
        return x * 10


@dataclass
class Predictor:

    model: Model

    def predict(self, batch: Sequence[Dict[str, int]]) -> Sequence[Dict[str, int]]:
        return [{"y": self.model(x["feature"])} for x in batch]

    @classmethod
    def from_artifacts_dict(cls, artifacts_dict):
        return cls(**artifacts_dict)


def save_model(artifact: Model, data_dir: Path) -> None:
    ...


def load_model(data_dir: Path, deps: Dict[str, Any]) -> Model:
    ...


def save_as_json(artifact: dict, data_dir: Path) -> None:
    ...


def load_from_json(data_dir: Path, deps) -> dict:
    ...


if __name__ == "__main__":

    model = Model()
    model_cfg: dict = {"layers": 2, "emb_size": 256}

    saver = (
        Saver(Predictor, code="", env_id="hm-ml-v1.1")  # noqa
        .set_info("hparams", {"lr": 0.01})
        .set_info("metrics", {"accuracy": 0.94})
        .add_artifact("model", model, save_model, load_model, deps=[alias("cfg", "model_cfg")])
        .add_artifact("model_cfg", model_cfg, save_as_json, load_from_json)
    )

    saver.savet("out/exp1")
        
# .add_info("dataset", {"source": "google.com"})
# .add_tags("simple", "cnn", "nlp", "binary-clf")
# .add_info("extra", {"annotator": {"name": "me"}})

# .with_code('.')
# .with_env(dockerfile="simple_classifier/Dockerfile")
# .with_meta(metrics={"accuracy": 0.94})
# .with_meta(hparams={"lr": 0.01})
# or

# saver = (   
#     Saver(Predictor)
#     .include_code(".")
#     .include_env(dockerfile="./dockerfile")
#     .include_meta(metrics={"accuracy": 0.94})
#     .add_artifact("model", model, save_model, load_model, deps=["cfg"])
#     .add_artifact("cfg", model_cfg, save_as_json, load_from_json)
# )

# saver = (
#     Saver(Predictor)
#     .add_artifact("model", placeholder(type_=Model, example=Model()), save_model, load_model, deps=[])
#     .dry_save("out/dir")
# )

# saver.save(
#     "out/dir", artifacts={"model": Model()}
# )

# or 

# (
#     saver
#     .meta(metrics={"F1": 0.93})
#     .add_artifact("model", Model())
#     .save("out/dir")
# )
# 
# .pack_artifact("model", model)
# .pack_artifact("cfg", model_cfg)
# # 0r

# saver = (   
#     Saver(Predictor)
#     .with_code(".")
#     .with_env(dockerfile="./dockerfile")
#     .with_meta(metrics={"accuracy": 0.94})
#     .add_artifact("model", model, save_model, load_model, deps=["cfg"])
#     .add_artifact("cfg", model_cfg, save_as_json, load_from_json)
# )

# saver.save("out/exp1")

# with saver.info(
#     metrics={"accuracy": 0.94},
#     dataset={"source": "supersecretdata.com"}
# ):
#     saver.save("out/exp1")

# (saver.info(metrics={"accuracy": 0.94}, dataset={"source": "supersecretdata.com"}).save()

# saver.save(
#     "out/exp2",
#     artifacts=[
#         Artifact("model", model, save_model, load_model, deps=["tokenizer", alias("cfg", "model_cfg")]),
#         Artifact("tokenizer", tokenizer, save_tok, load_tok, deps=[]),
#         Artifact("model_cfg", model_cfg, save_as_json, load_from_json)
#     ],
#     meta=ModelMeta()
# )


# def _save_model(_: Model, value: Model, data_dir: Path) -> None:
#     ...

# def _load_model(_: Model, cls: Type[Model], data_dir: Path, deps: dict) -> Model:
#     ...

#         Artifact("model", model, save_model, load_model, deps=["tokenizer", rename("model_cfg", "cfg")]),
# First attempt


# from sopredictable import save, load, Artifact

# save(
#     "out/exp2",
#     predictable_cls=Predictor,
#     artifacts=[
#         Artifact("model", model, save_model, load_model)
#     ],
#     meta={"hparams": {"lr": 0.2}, "metrics": {"precision": 0.53}},
#     dockerfile="./dockerfile"
# )

# load("out/exp2")


# # Second attempt

# sopredictable.save(
#     "out/exp1",
#     Predictor,
#     artifacts={
#         "model": model,
#         "cfg": sop.JSON(model_cfg),
#     },
#     deps = {
#         "model": ["cfg"]
#     },
#     meta={
#         "metrics": {"accuracy": 1.0},
#         "hparams": {"model": model_cfg}
#     },
#     dockerfile="./here/dockerfile"
# )

# predictor = sop.load("out/exp1")

# test_batch = [{"feature": 10}]
# print(predictor.predict(test_batch))

# # artifacts={
# #     "model": (model, save_model, load_model, ["cfg"]),
# #     "cfg": (model_cfg, save_as_json, load_from_json),
# # },


# # sop.save(
# #     tgt_dir="out/exp1",
# #     predictor_cls=Predictor,
# #     artifacts={"model": model}
# # )

# _ = Artifact("model", model, save_model, load_model, ["cfg"])
# _ = Artifact("cfg", model_cfg, save_as_json, load_from_json, ["cfg"])

# # from sopredictable import Saver, Artifact
# # saver = Saver(
# #     Predictor,
# #     artifacts=[
# #         Artifact("tokenizer", save_tok, load_tok),
# #         Artifact("cfg", save_json, load_json)
# #         Artifact("model", save_model, load_model, deps=["cfg", "tokenizer"]),
# #     ],
# # )
# # saver.save(
# #     dir_,
# #     artifacts={"model": model, "cfg": cfg, "tokenizer": tok}
# #     meta={"metrics": {"accuracy": 1.0}, "hparams": {"model": model_cfg}},
# # )
# # ...
# # later:
# # sopredictable.load(dir_)


# from sopredictable.core import Saver, Artifact, TrainInfo

# saver = Saver(
#     Predictor,
#     code=".",
#     dockerfile=""
# )

# saver.save(
#     "out/exp2",
#     artifacts=[
#         Artifact("model", model, save_model, load_model, deps=["model_cfg"]),
#         Artifact("model_cfg", model_cfg, save_as_json, load_from_json)
#     ],
#     meta=ModelMeta()
# )

# # ------


# # Attempt 3 (quite nice)


# from sopredictable.core import Saver, Loader, load_with_meta

# # saver = Saver(
# #     Predictor,
# #     code=".",
# #     dockerfile=""
# # )
# # saver.add_artifact("model", model, save_model, load_model, deps=["model_cfg"])


# from sopredictable import Bundle
# bundle = (
#    PredictorBundle(Predictor)
#    .pack_code("")
#    .pack_meta(meta)
#    .pack_env(env)
#    .add_artifact()
#    .add_artifact()
# )
#
# bundle.save("out/exp2")