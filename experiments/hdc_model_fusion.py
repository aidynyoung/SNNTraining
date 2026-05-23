"""HD-Glue Model Fusion Experiment. Fuses 3 model predictions via HDC consensus.
Based on Amrouch et al. Section V-C."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.hd_glue import HDGlue

def run_fusion(n_models=3, n_classes=4, dim=2000, n_train=200):
    glue = HDGlue(n_models, n_classes, dim=dim, mode="bipolar", seed=0)
    for _ in range(n_train):
        m = torch.randint(0, n_models, (1,)).item()
        c = torch.randint(0, n_classes, (1,)).item()
        glue.train_consensus(m, c)
    glue.normalize()
    correct = 0
    for _ in range(100):
        outs = torch.randn(n_models, n_classes).softmax(dim=1)
        true_class = torch.randint(0, n_classes, (1,)).item()
        outs[0, true_class] += 2.0
        pred = glue.predict(outs)
        if pred == true_class: correct += 1
    print(f"Consensus accuracy: {correct}/100")

if __name__ == "__main__":
    run_fusion()
