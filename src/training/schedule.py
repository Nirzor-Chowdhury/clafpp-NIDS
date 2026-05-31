from __future__ import annotations
from dataclasses import dataclass, field
@dataclass
class StageRecord:
    name: str
    description: str
@dataclass
class TrainingSchedule:
    config: dict
    stages: list[StageRecord]=field(default_factory=list)
    def __post_init__(self):
        self.stages=[StageRecord('stage_1_unsupervised','Train Autoencoder and GANomaly on normal traffic.'), StageRecord('stage_2_temporal','Train LSTM on sequence windows.'), StageRecord('stage_3_tabular','Train RandomForest and XGBoost.'), StageRecord('stage_4_pattern','Train CNN on feature patterns.'), StageRecord('stage_5_stacking','Train meta learner on validation predictions.'), StageRecord('stage_6_symbolic','Apply symbolic rules and compute CLAF++ scores.')]
    def to_dict(self): return {'stages':[s.__dict__ for s in self.stages]}
