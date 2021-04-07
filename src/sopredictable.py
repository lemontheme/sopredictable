from __future__ import annotations

import zipfile
import sys
import json
import functools as ft
import shutil
from pathlib import Path
import pickle
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
else:
    from typing_extensions import Protocol, TypedDict


PREDICTOR_CLS_FILE = "predictor_cls.pkl"
META_FILE = "meta.json"
CODE_ARCHIVE = "code.zip"
CONDA_FILE = "conda.yml"
ARTIFACTS_DIR = "artifacts"
ARTIFACT_DATA_DIR = "data"
ARTIFACT_CLS_FILE = "cls.pkl"
DEFAULT_ARTIFACT_FILE = "artifact.pkl"
DOCKERFILE_FILE = "dockerfile"
PICKLING_PROTOCOL = 4  # compatible with py37


@ft.singledispatch
def save_artifact(artifact: Any, data_dir: Path) -> None:
    try:
        (data_dir / DEFAULT_ARTIFACT_FILE).write_bytes(
            pickle.dumps(artifact, protocol=PICKLING_PROTOCOL)
        )
    except pickle.PicklingError:
        raise ValueError(f"Unable to pickle artifact {artifact}")


@ft.singledispatch
def load_artifact(artifact_cls: Any, data_dir: Path) -> Any:
    print(f"loading {artifact_cls=}")
    try:
        return pickle.loads((data_dir / DEFAULT_ARTIFACT_FILE).read_bytes())
    except pickle.UnpicklingError:
        raise ValueError(f"Unable to pickle artifact of type {artifact_cls}")


class ModelMeta(TypedDict, total=False):
    hparams: Optional[dict]
    metrics: Optional[Dict[str, Union[float, Dict[str, float]]]]
    tags: Optional[Sequence[str]]


T = TypeVar("T")


@runtime_checkable
class Predictable(Protocol):
    def predict(self, batch: Sequence[Any]) -> Sequence[Any]:
        ...

    @classmethod
    def from_artifacts_dict(cls: Type[T], artifacts_dict: Dict[str, Any]) -> T:
        ...


def save(
    dir_path: Union[str, Path],
    predictor_cls: Type[Predictable],
    artifacts_dict: Dict[str, Any] = None,
    code: Path = None,
    meta: ModelMeta = None,
    conda_env: Path = None,
    dockerfile: Path = None,
) -> None:
    """
    Args:
        path: DIRECTORY
        predictor_cls: ...
        artifacts_dict: ...
        code (optional): path to source code dir.
        meta: ...
        conda_env (optional): path to `conda.yml` file.
    """
    # Prepare target directory
    dir_path = Path(dir_path)
    if dir_path.exists():
        if sum(1 for _ in dir_path.iterdir()) > 0:
            raise FileExistsError(
                f"dir '{dir_path}' already exists. Caller should ensure it is removed."
            )
    else:
        dir_path.mkdir(parents=True)
    (dir_path / PREDICTOR_CLS_FILE).write_bytes(pickle.dumps(predictor_cls))
    artifacts_dir = dir_path / ARTIFACTS_DIR
    artifacts_dir.mkdir()
    for artifact_name, artifact in artifacts_dict.items() or ():
        artifact_dir = artifacts_dir / artifact_name
        artifact_dir.mkdir()
        artifact_cls = type(artifact)
        (artifact_dir / ARTIFACT_CLS_FILE).write_bytes(pickle.dumps(artifact_cls))
        artifact_data_dir = artifact_dir / ARTIFACT_DATA_DIR
        artifact_data_dir.mkdir()
        save_artifact(artifact, artifact_dir / ARTIFACT_DATA_DIR)

    # Model metadata
    (dir_path / META_FILE).write_text(json.dumps(meta or {}))
    # Code
    if code is not None:
        if not code.is_dir():
            raise ValueError("`code` does not point to a directory.")
        else:
            with zipfile.ZipFile((dir_path / CODE_ARCHIVE)) as zip_fp:
                files = (f for f in code.glob("**") if f.suffix != "pyc")
                for f in files:
                    zip_fp.write(filename=f, arcname=f.relative_to(code))
    if conda_env:
        shutil.copy(src=conda_env, dst=(dir_path / CONDA_FILE))
    if dockerfile:
        shutil.copy(src=dockerfile, dst=(dir_path / CONDA_FILE))


def load(dir_path: Union[str, Path]) -> Predictable:
    dir_path = Path(dir_path)
    return _load_predictor(dir_path)


def load_with_meta(dir_path: Union[str, Path]) -> Tuple[Predictable, ModelMeta]:
    dir_path = Path(dir_path)
    return _load_predictor(dir_path), _load_meta(dir_path)


def _load_predictor(dir_path: Path) -> Predictable:
    if not dir_path.exists():
        raise ValueError(f"`{dir_path}` does not exist.")
    if (code_archive := (dir_path / CODE_ARCHIVE)).exists():
        sys.path.append(str(code_archive))
    artifacts_dict = {}
    for artifact_dir in (dir_path / ARTIFACTS_DIR).iterdir():
        if artifact_dir.name.startswith("."):
            continue
        artifact_cls = pickle.loads((artifact_dir / ARTIFACT_CLS_FILE).read_bytes())
        artifact = load_artifact(artifact_cls, artifact_dir / ARTIFACT_DATA_DIR)
        artifact_name = artifact_dir.name
        artifacts_dict[artifact_name] = artifact
    predictor_cls: Type[Predictable] = pickle.loads(
        (dir_path / PREDICTOR_CLS_FILE).read_bytes()
    )
    predictor: Predictable = predictor_cls.from_artifacts_dict(artifacts_dict)
    return predictor


def _load_meta(dir_path: Path) -> ModelMeta:
    meta = json.loads((dir_path / META_FILE).read_text())
    return meta


__all__ = ["save", "load", "save_artifact", "load_artifact", "ModelMeta", "Predictable"]

__version__ = "0.1.0"
