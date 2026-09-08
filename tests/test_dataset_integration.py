import pandas as pd
import pytest

from app.analyzers import REGISTRY
from app.core.engine import Engine
from app.core.spec import load_spec
from experiments.common import DOCUMENT_SPEC, LABEL_COLUMNS, image_files
from experiments.evaluate import evaluate_dataset, requirement_ids
from experiments.metrics import error_rates

pytestmark = pytest.mark.dataset

SAMPLE = 5


def test_pipeline_runs_on_valid_portraits(data_valid_dir, backend):
    paths = image_files(data_valid_dir)[:SAMPLE]
    labels = pd.DataFrame(
        [(p.name, p.name, "clean", "", "test", "") for p in paths], columns=list(LABEL_COLUMNS)
    )
    engine = Engine(load_spec(DOCUMENT_SPEC, REGISTRY))
    frame = evaluate_dataset(labels, data_valid_dir, engine, backend)
    assert len(frame) == SAMPLE
    assert set(requirement_ids(frame.columns)) == set(engine.spec.ids)
    assert frame["conclusion_ok"].all()
    assert frame["elapsed_ms"].median() < 2000
    rates = error_rates(frame)
    assert 0.0 <= rates.frr <= 1.0
