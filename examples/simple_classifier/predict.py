from dataclasses import dataclass
from typing import Dict, Sequence

import sopredictable


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


sopredictable.save(
    "out/exp1", Predictor, {"model": Model()}, meta={"metrics": {"accuracy": 1.0}}
)

predictor = sopredictable.load("out/exp1")

test_batch = [{"feature": 10}]
print(predictor.predict(test_batch))


