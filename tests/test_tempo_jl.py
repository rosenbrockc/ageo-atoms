"""Tests for tempo_jl."""\n\nimport pytest\nimport numpy as np\nimport icontract\nfrom ageoa.tempo_jl.atoms import graph_time_scale_management\nfrom ageoa.tempo_jl.atoms import high_precision_duration\n\ndef test_graph_time_scale_management_positive():
    with pytest.raises(NotImplementedError):
        graph_time_scale_management(np.array([1.0]))

def test_graph_time_scale_management_precondition():
    with pytest.raises(icontract.ViolationError):
        graph_time_scale_management(None)

def test_high_precision_duration_positive():
    with pytest.raises(NotImplementedError):
        high_precision_duration(np.array([1.0]))

def test_high_precision_duration_precondition():
    with pytest.raises(icontract.ViolationError):
        high_precision_duration(None)

