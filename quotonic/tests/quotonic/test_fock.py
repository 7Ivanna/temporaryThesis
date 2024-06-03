from quotonic.fock import basis, getDim


def test_getDim():
    assert getDim(2, 4) == 10


def test_basis():
    assert basis(2, 2) == [[2, 0], [1, 1], [0, 2]]
    assert basis(2, 4) == [
        [2, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 2, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 2, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 2],
    ]
