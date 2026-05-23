"""HDC Robustness Experiment: SNN spike encoding + hardware error masking.
Tests Associative Memory accuracy under bit-flip errors with zero/sign/word masking.
Based on Amrouch et al. Section III-A."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from models.hdc import HDCEncoder, corrupt_hv, mask_zero, mask_sign, mask_word, AssocMemory
from data.synthetic import bci_velocity_stream

def angle_to_label(vel):
    a = torch.atan2(vel[1], vel[0]).item()
    return int(((a + 3.1416) / (2 * 3.1416)) * 8) % 8

def run_experiment(dim=4096, T_train=1000, T_test=500, error_rates=[0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = HDCEncoder(input_size=100, n_classes=8, dim=dim, mode="bipolar", device=device, seed=0)
    train_stream = list(bci_velocity_stream(T=T_train, input_size=100, seed=0))
    for x, y in train_stream:
        enc.train_step(x.to(device), angle_to_label(y))
    enc.finalize()

    test_stream = list(bci_velocity_stream(T=T_test, input_size=100, seed=42))
    test_hvs = [(enc.encode(x.to(device)), angle_to_label(y)) for x, y in test_stream]

    results = {}
    for masking in ["none", "zero", "signbit", "word"]:
        accs = {}
        for rate in error_rates:
            correct = 0
            for hv, label in test_hvs:
                c = corrupt_hv(hv, rate, mode="bipolar", etype="flip")
                emask = (c != hv)
                if masking == "none": pred = enc.memory.predict(c)
                elif masking == "zero": pred = enc.memory.predict(mask_zero(c, emask))
                elif masking == "signbit": pred = enc.memory.predict(mask_sign(c, emask, "bipolar"))
                elif masking == "word": pred = enc.memory.predict(mask_word(c, emask, word_size=8))
                if pred == label: correct += 1
            accs[rate] = correct / len(test_hvs)
        results[masking] = accs
        print(f"Masking={masking}: {accs}")
    return results

if __name__ == "__main__":
    print("HDC Robustness under hardware bit-flip errors")
    run_experiment()
