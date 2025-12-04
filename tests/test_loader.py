import pandas as pd
import pytest
from src.data_loader import InsuranceDataLoader

# Create a dummy CSV for testing
@pytest.fixture
def create_dummy_csv(tmpdir):
    df = pd.DataFrame({
        'TransactionMonth': ['2014-01-01', '2014-02-01'],
        'TotalPremium': [1000, 2000],
        'TotalClaims': [0, 500]
    })
    file = tmpdir.join("test_data.csv")
    df.to_csv(file, index=False)
    return str(file)

def test_data_loader(create_dummy_csv):
    loader = InsuranceDataLoader(create_dummy_csv)
    df = loader.load_data()
    assert df is not None
    assert not df.empty
    assert 'TotalPremium' in df.columns