from app.core.hints import hint_context, hint_placeholders


def test_direction_words_follow_sign():
    context = hint_context({"yaw": 17.3, "pitch": -4.0, "dx": 0.12, "dy": -0.05})
    assert context["yaw_dir"] == "влево"
    assert context["yaw_abs"] == 17.3
    assert context["pitch_dir"] == "поднимите подбородок"
    assert context["pitch_abs"] == 4.0
    assert context["dx_dir"] == "левее"
    assert context["dy_dir"] == "ниже"


def test_negative_yaw_and_asymmetry():
    context = hint_context({"yaw": -3.0, "asymmetry": 0.2, "roll": 5.0})
    assert context["yaw_dir"] == "вправо"
    assert context["asymmetry_dir"] == "справа"
    assert context["roll_dir"] == "против часовой стрелки"


def test_unsigned_keys_produce_nothing():
    assert hint_context({"count": 2, "min_ear": 0.1}) == {}


def test_hint_placeholders():
    assert hint_placeholders(("yaw", "pitch", "roll")) == {
        "yaw_dir",
        "yaw_abs",
        "pitch_dir",
        "pitch_abs",
        "roll_dir",
        "roll_abs",
    }
    assert hint_placeholders(("count",)) == set()
