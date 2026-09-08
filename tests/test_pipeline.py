import numpy as np

from app.analyzers.registry import MeasurerSpec
from app.core.engine import Engine
from app.core.pipeline import check_image
from app.core.report import Measurement, Status
from app.core.spec import parse_spec
from app.imageio import write_bgr
from tests.helpers import FakeBackend, box_landmarks, requirement, spec_data


def width_measurer(ctx):
    return Measurement.of("resolution.size", {"width": ctx.width, "faces": len(ctx.faces)})


def test_check_image_reads_file_and_times_it(tmp_path):
    path = tmp_path / "снимок.jpg"
    write_bgr(path, np.full((40, 60, 3), 128, dtype=np.uint8), quality=90)
    registry = {
        "resolution.size": MeasurerSpec("resolution.size", ("width", "faces"), width_measurer)
    }
    spec = parse_spec(
        spec_data(
            requirement("wide", "resolution.size", {"op": "ge", "value": 50}),
            requirement("one_face", "resolution.size", {"op": "eq", "key": "faces", "value": 1}),
        ),
        registry,
    )
    backend = FakeBackend([box_landmarks(10, 10, 30, 30, 60, 40)], np.zeros((40, 60), np.float32))
    result = check_image(path, Engine(spec), backend)
    assert result.report.image == "снимок.jpg"
    assert result.report.accepted
    assert result.report.by_id("wide").status is Status.PASS
    assert result.report.by_id("one_face").measured == {"faces": 1}
    assert result.elapsed_ms > 0
