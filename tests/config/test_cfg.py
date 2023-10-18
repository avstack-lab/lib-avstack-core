from avstack.config import Config


def test_load_base_config():
    fname = "tests/config/base_cfg.py"
    cfg = Config.fromfile(fname)
    assert cfg.element_0 == 1
    assert cfg.filename == fname


def test_load_derived_config():
    fname = "tests/config/derived_cfg.py"
    cfg = Config.fromfile(fname)
    assert cfg.element_0 == 2
    assert cfg.element_2.field_1 == "200"
    assert cfg.element_2.field_2.subfield_1 == 21
    assert cfg.filename == fname


def test_list_in_config():
    fname = "tests/config/base_cfg.py"
    cfg = Config.fromfile(fname)
    assert len(cfg.element_3) == 3
    assert isinstance(cfg.element_3[0], dict)