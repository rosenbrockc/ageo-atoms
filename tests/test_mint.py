"""Tests for mint."""\n\nimport pytest\nimport numpy as np\nimport icontract\nfrom ageoa.mint.atoms import axial_attention\nfrom ageoa.mint.atoms import rotary_positional_embeddings\n\ndef test_axial_attention_positive():
    with pytest.raises(NotImplementedError):
        axial_attention(np.array([1.0]))

def test_axial_attention_precondition():
    with pytest.raises(icontract.ViolationError):
        axial_attention(None)

def test_rotary_positional_embeddings_positive():
    with pytest.raises(NotImplementedError):
        rotary_positional_embeddings(np.array([1.0]))

def test_rotary_positional_embeddings_precondition():
    with pytest.raises(icontract.ViolationError):
        rotary_positional_embeddings(None)

