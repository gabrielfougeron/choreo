from pytest_pyodide import run_in_pyodide
@run_in_pyodide
def test_add(selenium):
    assert 1 + 1 == 2
    assert 1 + 2 == 2