import torch
import torchtuples as tt
from pycox.models import CoxPH as DeepSurv
from pycox.preprocessing.label_transforms import LabTransCoxTime
import shap

print("=" * 60)
print("SYSTEM CHECK - MacBook M1 Pro")
print("=" * 60)

# Check PyTorch
print(f"✓ PyTorch version: {torch.__version__}")

# Check MPS availability
if torch.backends.mps.is_available():
    print("✓ MPS (Metal) acceleration: AVAILABLE 🚀")
    print("  → Neural networks will use GPU acceleration!")
    
    # Test MPS
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"  → Test tensor on MPS: {x}")
else:
    print("⚠ MPS acceleration: NOT AVAILABLE")
    print("  → Will use CPU (still fast on M1)")

# Check other libraries
print("✓ torchtuples: OK")
print("✓ pycox: OK")  
print("✓ SHAP: OK")
print("=" * 60)
print("\n Tutto pronto per training accelerato su M1!")