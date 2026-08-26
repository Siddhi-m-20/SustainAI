import numpy
import pandas
import sklearn
import plotly
import streamlit


def test_environment():
    assert numpy.__version__
    assert pandas.__version__
    assert sklearn.__version__
    assert plotly.__version__
    assert streamlit.__version__