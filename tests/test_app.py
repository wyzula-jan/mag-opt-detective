from mag_opt_detective import app


def test_smoke_test_passes(qapp):
    assert app.main(["mag-opt-detective", "--smoke-test"]) == 0


def test_qt_arguments_are_ignored():
    args = app.parse_args(["mag-opt-detective", "-platform", "offscreen", "--smoke-test"])
    assert args.smoke_test
