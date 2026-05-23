import torch
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rsnn import RSNN
from models.readout import Readout
from models.hebbian import DualHebbian
from models.lif import LIFLayer

def test_rsnn_initialization():
    """Test RSNN initialization and basic functionality."""
    device = torch.device('cpu')
    rsnn = RSNN(10, 20, device=device)
    
    assert rsnn.hidden_size == 20
    assert rsnn.W_in.shape == (20, 10)
    assert rsnn.W_rec.shape == (20, 20)
    assert rsnn.prev_spikes.shape == (20,)
    
def test_rsnn_forward():
    """Test RSNN forward pass."""
    device = torch.device('cpu')
    rsnn = RSNN(10, 20, device=device)
    x = torch.randn(10)
    
    spikes = rsnn.forward(x)
    assert spikes.shape == (20,)
    assert torch.all(spikes >= 0) and torch.all(spikes <= 1)
    
def test_lif_layer():
    """Test LIF neuron layer."""
    device = torch.device('cpu')
    lif = LIFLayer(50, device=device)
    
    current = torch.randn(50)
    spikes = lif.step(current)
    
    assert spikes.shape == (50,)
    assert torch.all(spikes >= 0) and torch.all(spikes <= 1)
    
def test_readout_layer():
    """Test readout layer."""
    device = torch.device('cpu')
    readout = Readout(50, 2, device=device)
    spikes = torch.randn(50)
    
    output = readout.forward(spikes)
    assert output.shape == (2,)
    
def test_hebbian_update():
    """Test dual-timescale Hebbian learning."""
    device = torch.device('cpu')
    hebbian = DualHebbian((50, 50), device=device)
    
    pre = torch.randn(50)
    post = torch.randn(50)
    
    eligibility = hebbian.update(pre, post)
    assert eligibility.shape == (50, 50)
    
    # Check that eligibility traces are updated
    assert torch.norm(hebbian.e_fast) > 0
    assert torch.norm(hebbian.e_slow) > 0

if __name__ == "__main__":
    pytest.main([__file__])
