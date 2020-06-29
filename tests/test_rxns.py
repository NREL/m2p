from m2p import PolyMaker
import pytest

pm = PolyMaker()

@pytest.fixture
def vinyl():
	return pm.polymerize("C=C",DP=10,mechanism="vinyl").polymer[0]

def test_vinyl(vinyl):
	smile_vinyl = vinyl
	assert smile_vinyl == 'CCCCCCCCCCCCCCCCCCCC'

