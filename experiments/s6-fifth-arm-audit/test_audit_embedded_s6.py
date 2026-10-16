import ast

import pytest

from audit_embedded_s6 import assignment


def test_assignment_accepts_one_literal() -> None:
    tree = ast.parse("VALUE = {'a': 1}\n")
    assert assignment(tree, "VALUE") == {"a": 1}


def test_assignment_rejects_duplicates() -> None:
    tree = ast.parse("VALUE = 1\nVALUE = 2\n")
    with pytest.raises(RuntimeError, match="expected one literal assignment"):
        assignment(tree, "VALUE")


def test_assignment_supports_destructuring() -> None:
    tree = ast.parse("LEFT, RIGHT = 'a', 2\n")
    assert assignment(tree, "LEFT") == "a"
    assert assignment(tree, "RIGHT") == 2
