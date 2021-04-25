from __future__ import annotations

import inspect
import copy
import dataclasses
from dataclasses import dataclass
import os
import typing
import zipfile
import sys
import json
import functools as ft
import shutil
from pathlib import Path
from collections import deque
import pickle
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
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
ARTIFACT_SAVE_FN_FILE = "save_fn.pkl"
ARTIFACT_LOAD_FN_FILE = "load_fn.pkl"
DEFAULT_ARTIFACT_FILE = "artifact.pkl"
DOCKERFILE_FILE = "Dockerfile"
REQUIREMENTS_TXT_FILE = "requirements.txt"
PICKLING_PROTOCOL = 4  # compatible with py37


PathLike = typing.Union[str, os.PathLike, Path]
ArtifactName = str

JsonSerializable = Union[
    bool, None, Dict[str, "JsonSerializable"], List["JsonSerializable"], int, float, str
]
JsonDict = Dict[str, JsonSerializable]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class Predictable(Protocol):
    def predict(self, batch: Sequence[Any]) -> Sequence[Any]:
        ...

    @classmethod
    def from_artifacts_dict(cls: Type[T], artifacts_dict: Dict[str, Any]) -> T:
        ...


@runtime_checkable
class ArtifactSaver(Protocol[T_contra]):
    def __call__(self, artifact: T_contra, data_dir: Path) -> None:
        ...


@runtime_checkable
class ArtifactLoader(Protocol[T_co]):
    def __call__(self, data_dir: Path, deps: Dict[str, Any]) -> T_co:
        ...


def save_to_pickle(artifact: Any, data_dir: Path) -> None:
    try:
        (data_dir / DEFAULT_ARTIFACT_FILE).write_bytes(
            pickle.dumps(artifact, protocol=PICKLING_PROTOCOL)
        )
    except pickle.PicklingError:
        raise ValueError(f"Unable to pickle {artifact}")


def load_from_pickle(data_dir: Path, deps: Dict[str, Any]) -> Any:
    _ = deps
    pickle_fpath = data_dir / DEFAULT_ARTIFACT_FILE
    try:
        return pickle.loads(pickle_fpath.read_bytes())
    except pickle.UnpicklingError:
        raise ValueError(f"Unable to load pickle from {pickle_fpath}")


@dataclass(frozen=True)
class _Alias:
    name: str
    referent: str


def alias(
    name: str,
    referent: str,
):
    """
    Args:
        name: alias name, e.g. name of key in dict passed to `deps` parameter of
          functions implementing `ArtifactLoader` protocol.
        referent: name that alias refers to, e.g. the artifact name registered via
          `Saver.add_artifact()`.
    """
    return _Alias(name, referent)


def placeholder():
    raise NotImplementedError


@dataclass(frozen=True)
class _PredictorInfo:
    hparams: Optional[JsonDict] = None
    metrics: Optional[JsonDict] = None
    dataset: Optional[JsonDict] = None
    task: Optional[JsonDict] = None
    extra: Optional[JsonDict] = None
    _tags: Optional[Sequence[str]] = None


class _ArtifactInfo(TypedDict):
    name: str
    cls: str
    save: str
    load: str
    deps: list


@dataclass(frozen=True)
class _AllMeta:
    _info: _PredictorInfo
    _created_at: str
    _sopredictable_version: str
    _python_version: str
    _os: str
    _artifacts: List[_ArtifactInfo]
    _artifact_load_order: List[str]


@dataclass(frozen=True)
class _ArtifactRecord:
    name: str
    value: Any
    save: ArtifactSaver
    load: ArtifactLoader
    deps: Sequence[_LoadDep] = ()

    def __post_init__(self):
        assert self.value is not None
        assert isinstance(self.name, str) and len(self.name) > 0
        assert callable(self.save)
        assert callable(self.load)
        self._validate_deps()

    def _validate_deps(self):
        seen_names, seen_aliases = set(), set()
        for dep in self.deps:
            if dep.artifact_name in seen_names or dep.load_time_name in seen_aliases:
                raise ValueError(
                    f"`deps` refers to the same artifact dependency more than once."
                )
            else:
                seen_names.add(dep.artifact_name)
                seen_aliases.add(dep.load_time_name)

    def to_dict(self) -> _ArtifactInfo:
        return {
            "name": self.name,
            "cls": type(self.value).__qualname__,
            "save": _qualname_from_user_callable(self.save),
            "load": _qualname_from_user_callable(self.load),
            "deps": [dataclasses.asdict(d) for d in self.deps],
        }


def _qualname_from_user_callable(c: Callable) -> str:
    if inspect.isfunction(c):
        return c.__qualname__
    elif callable(c) and not inspect.isclass(c):
        return type(c).__qualname__
    else:
        raise ValueError


@dataclass(frozen=True)
class _Environment:
    dockerfile: Optional[Path] = None
    requirements_txt: Optional[Path] = None
    conda_env: Optional[Path] = None
    env_id: Optional[str] = None


@dataclass(frozen=True)
class _LoadDep:
    artifact_name: str
    load_time_name: Optional[str] = None

    def __post_init__(self):
        if self.load_time_name is None:
            object.__setattr__(self, "load_time_name", self.artifact_name)


# Saving
# ------

InfoField = Literal["hparams", "metrics", "dataset", "extra"]


class Saver:

    __slots__ = ("_predictor_cls", "_code", "_info", "_env", "_artifacts", "_tags")

    _predictor_cls: Type[Predictable]
    _code: Optional[Path]
    _info: _PredictorInfo
    _env: _Environment
    _artifacts: Tuple[_ArtifactRecord, ...]
    _options: Dict[str, bool]

    def __init__(
        self,
        predictor_cls: Type[Predictable],
        code: PathLike = None,
        tags: List[str] = None,
        dockerfile: PathLike = None,
        conda_env: PathLike = None,
        requirements: PathLike = None,
        env_id: str = None,
    ) -> None:
        self._predictor_cls = predictor_cls
        self._env = _Environment(
            dockerfile=Path(dockerfile) if dockerfile else None,
            conda_env=Path(conda_env) if conda_env else None,
            requirements_txt=Path(requirements) if requirements else None,
            env_id=env_id,
        )
        self._code = Path(code) if code else None
        self._info = _PredictorInfo()
        self._artifacts = ()

    def set_info(
        self,
        field: InfoField,
        value: JsonDict,
        *,
        on_conflict: Literal["raise", "merge", "replace"] = "raise",
    ) -> Saver:
        if field not in typing.get_args(InfoField):
            raise ValueError(f"No such field '{field}' to assign to.")
        current_info = self._info
        # If conflict (i.e. field already has value)
        if getattr(current_info, field) is not None:
            if on_conflict == "raise":
                raise RuntimeError(
                    f"Field '{field}' already has a value. Consider setting `on_conflict`"
                    f"to 'merge' or 'replace'."
                )
            elif on_conflict == "replace":
                pass
            elif on_conflict == "merge":
                current_value = getattr(current_info, field)
                value = _update_dict_by_merge(old=current_value, new=value)
            else:
                raise ValueError(
                    f"Unrecognized conflict resolution method '{on_conflict}'."
                )
        new_info = dataclasses.replace(current_info, **{field: value})
        return self._replace(_info=new_info)

    def add_artifact(
        self,
        name: str,
        artifact: T,
        save: ArtifactSaver[T] = save_to_pickle,
        load: ArtifactLoader[T] = load_from_pickle,
        deps: Sequence[Union[str, _Alias]] = (),
    ) -> Saver:
        if name in (ar.name for ar in self._artifacts):
            raise ValueError(f"Artifact name '{name}' already in use.")
        deps_ = []
        for dep in deps:
            if isinstance(dep, str):
                deps_.append(_LoadDep(dep))
            elif isinstance(dep, _Alias):
                deps_.append(_LoadDep(dep.referent, load_time_name=dep.name))
            else:
                raise TypeError(
                    f"`deps` member '{dep}' is neither a string nor an `_Alias`"
                )
        artifact_record = _ArtifactRecord(name, artifact, save, load, deps_)
        new_obj: Saver = self._replace(
            _artifacts=tuple((*self._artifacts, artifact_record))
        )
        return new_obj

    def save(self, path: PathLike, dry=False) -> None:
        if dry:
            raise NotImplementedError
        _save(Path(path), self)

    def _replace(self, **changes: Any) -> Saver:
        copy_of_self = copy.copy(self)
        for attr_name, value in changes.items():
            setattr(copy_of_self, attr_name, value)
        return copy_of_self

    def _check_correctness(self) -> None:
        ...


# Loading

# fmt: off
@typing.overload
def load(path: PathLike, with_info: Literal[False]) -> Predictable:
    ...
@typing.overload
def load(
    path: PathLike, with_info: Literal[True]
) -> Tuple[Predictable, _PredictorInfo]:
    ...
def load(
    path: PathLike, with_info: bool = False
) -> Union[Predictable, Tuple[Predictable, _PredictorInfo]]:
    path_ = Path(path)
    predictor, meta = _load(path_)
    if with_info:
        return predictor, meta._info
    else:
        return predictor
# fmt: on


def _save(path: Path, saver: Saver) -> None:
    """
    Args:
        path: DIRECTORY
        saver (Saver): ...
    """
    # Prepare target directory
    dir_path = path
    if dir_path.exists():
        if sum(1 for _ in dir_path.iterdir()) > 0:
            raise FileExistsError(
                f"dir '{dir_path}' already exists. Caller should ensure it is removed."
            )
    else:
        dir_path.mkdir(parents=True)
    (dir_path / PREDICTOR_CLS_FILE).write_bytes(pickle.dumps(saver._predictor_cls))
    artifacts_dir = dir_path / ARTIFACTS_DIR
    artifacts_dir.mkdir()
    for artifact_record in saver._artifacts:
        artifact_dir = artifacts_dir / artifact_record.name
        artifact_dir.mkdir()
        artifact_data_dir = artifact_dir / ARTIFACT_DATA_DIR
        artifact_data_dir.mkdir()
        (artifact_dir / ARTIFACT_SAVE_FN_FILE).write_bytes(
            pickle.dumps(artifact_record.save)
        )
        (artifact_dir / ARTIFACT_LOAD_FN_FILE).write_bytes(
            pickle.dumps(artifact_record.load)
        )
        artifact_record.save(artifact_record, artifact_dir / ARTIFACT_DATA_DIR)

    # Code
    code_path: Optional[Path] = saver._code
    if not code_path:
        module = inspect.getmodule(saver._predictor_cls)
        if module:
            module_fpath = module.__file__
            code_path = Path(module_fpath) if module_fpath != "__main__" else None
    if code_path:
        with zipfile.ZipFile((dir_path / CODE_ARCHIVE), mode="w") as zip_fp:
            if code_path.is_dir():
                files = (f for f in code_path.glob("**") if f.suffix != "pyc")
                for f in files:
                    zip_fp.write(filename=f, arcname=f.relative_to(code_path))
            else:
                f = code_path
                zip_fp.write(filename=f, arcname=f.name)

    # Environment
    for env_attr, out_path in [
        ("conda_env", CONDA_FILE),
        ("dockerfile", DOCKERFILE_FILE),
        ("requirements_txt", REQUIREMENTS_TXT_FILE),
    ]:
        attr_val = getattr(saver._env, env_attr)
        src_path = Path(attr_val).resolve() if attr_val else None
        if src_path:
            shutil.copy(src=src_path, dst=(dir_path / out_path))

    # Finally, all metadata.
    (
        (dir_path / META_FILE).write_text(
            json.dumps(dataclasses.asdict(_generate_meta(saver)), indent=2)
        )
    )


def _generate_meta(saver: Saver):
    from datetime import datetime
    from sopredictable.about import __version__
    import platform

    info = saver._info
    meta = _AllMeta(
        _info=info,
        _created_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
        _sopredictable_version=__version__,
        _python_version=platform.python_version(),
        _os=platform.platform(),
        _artifacts=[artifact.to_dict() for artifact in saver._artifacts],
        _artifact_load_order=_compute_artifact_load_order(saver._artifacts),
    )
    print(meta)
    return meta


def _dependency_graph_from_artifacts(
    artifacts: Sequence[_ArtifactRecord],
) -> Dict[ArtifactName, Sequence[ArtifactName]]:
    return {
        artifact.name: [dep.artifact_name for dep in artifact.deps]
        for artifact in artifacts
    }


def _compute_artifact_load_order(
    artifacts: Sequence[_ArtifactRecord],
) -> List[ArtifactName]:
    dependency_graph = _dependency_graph_from_artifacts(artifacts)
    # TODO: handle cycles
    load_order = _iterative_topological_sort(dependency_graph)
    return load_order


def _load(dir_path) -> Tuple[Predictable, _AllMeta]:

    if not dir_path.exists():
        raise ValueError(f"`{dir_path}` does not exist.")

    code_archive = dir_path / CODE_ARCHIVE
    if code_archive.exists():
        sys.path.append(str(code_archive))

    meta: _AllMeta = _load_meta(dir_path)
    artifact_load_order = meta._artifact_load_order

    _artifact2deps: Dict[ArtifactName, List[_LoadDep]] = {
        atfct["name"]: [
            _LoadDep(dep["artifact_name"], dep["_load_time_name"])
            for dep in atfct["deps"]
        ]
        for atfct in meta._artifacts
    }
    artifacts_dict = {}
    for artifact_name in artifact_load_order:
        artifact_dir = dir_path / artifact_name
        load_fn = pickle.loads((artifact_dir / ARTIFACT_LOAD_FN_FILE).read_bytes())
        deps_arg = {
            dep.load_time_name: artifacts_dict[dep.artifact_name]
            for dep in _artifact2deps[artifact_name]
        }
        artifact_data_dir = artifact_dir / ARTIFACT_DATA_DIR
        artifact = load_fn(artifact_data_dir, deps=deps_arg)
        artifacts_dict[artifact_name] = artifact

    predictor_cls: Type[Predictable] = pickle.loads(
        (dir_path / PREDICTOR_CLS_FILE).read_bytes()
    )
    predictor: Predictable = predictor_cls.from_artifacts_dict(artifacts_dict)
    return predictor, meta


def _load_meta(dir_path: Path) -> _AllMeta:
    meta = json.loads((dir_path / META_FILE).read_text())
    return _AllMeta(**meta)


def _iterative_topological_sort(graph: Dict[Hashable, Sequence[Hashable]]):
    """
    Stolen from https://stackoverflow.com/a/47234034/15632334 with minor modifications
    after giving up on implementing it myself.

    Example:
        >>> G1 = {
        ...   "a": ["b", "c"],
        ...   "b": ["d"],
        ...   "c": ["d"],
        ...   "d": []
        ... }
        >>> iterative_topological_sort(G1)
        ['d', 'c', 'b', 'a']
        >>> G1 = {
        ...     "a": ["b", "c"],
        ...     "b": [],
        ...     "c": ["b"]
        ... }  # top. sort: "b", "c", "a"
        >>> iterative_topological_sort(G2)
        ['b', 'c', 'a']
    """
    seen = set()
    stack = []
    order = []
    q = list(graph)
    while q:
        node = q.pop()
        if node not in seen:
            seen.add(node)  # no need to append to path any more
            q.extend(graph[node])
            while stack and node not in graph[stack[-1]]:  # new stuff here!
                order.append(stack.pop())
            stack.append(node)
    order.extend(stack)
    return order


def _update_dict_by_merge(old: dict, new: dict) -> dict:
    result = copy.deepcopy(old)
    stack: list = [(result, new)]
    while stack:
        old, new = stack.pop()
        for k in new:
            if k in old and isinstance(old[k], dict) and isinstance(new[k], dict):
                stack.append((old[k], new[k]))
            else:
                old[k] = new[k]
    return result


__all__ = ["save", "load", "Predictable"]
