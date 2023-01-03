def check_xor_for_none(a, b):
    assert check_xor(a is None, b is None), "Can only pass in one of these inputs"


def check_xor(a, b):
    return (a or b) and (not a or not b)
