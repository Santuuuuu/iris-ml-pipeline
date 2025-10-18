"""
This test will ALWAYS pass
"""
def test_one_plus_one():
    assert 1 + 1 == 2

def test_true_is_true():
    assert True

def test_python_works():
    import sys
    assert sys.version_info.major == 3
