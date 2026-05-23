"""python -m snntraining entry point."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def cmd_demo(args):
    import torch
    from hdc.hdcc_compiler import HDCCClassifier

    steps = args.steps
    dim = 1000
    n_features = 10
    n_classes = 4

    print()
    print("SNNTraining demo — online HDC learning, no backpropagation")
    print(f"steps={steps}  dim={dim}  features={n_features}  classes={n_classes}")
    print()

    model = HDCCClassifier(
        n_features=n_features,
        n_classes=n_classes,
        dim=dim,
        n_projections=4,
        mode="binary",
        learning_rate=0.1,
    )

    torch.manual_seed(42)
    samples_per_class = max(1, steps // n_classes)

    print(f"Training: {samples_per_class} samples/class ({samples_per_class * n_classes} total)")
    for cls in range(n_classes):
        for _ in range(samples_per_class):
            x = torch.randn(n_features) * 0.3 + cls * 0.5
            model.train_step(x, cls, predict_first=False)
    model.renormalize()

    n_test = 200
    correct = 0
    for cls in range(n_classes):
        for _ in range(n_test // n_classes):
            x = torch.randn(n_features) * 0.3 + cls * 0.5
            pred, _ = model.predict(x)
            if pred == cls:
                correct += 1

    accuracy = correct / n_test
    energy = model.estimate_energy()

    print(f"Accuracy:          {accuracy:.1%}")
    print(f"Energy/inference:  {energy['total_energy_nj_per_inference']} nJ")
    print(f"vs Transformer:    {energy['energy_ratio_vs_transformer']}")
    print()
    print("Done.")


def cmd_train(args):
    from train import main as train_main
    train_main()


def main():
    parser = argparse.ArgumentParser(
        prog="snntraining",
        description="Train spiking neural networks without backpropagation.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p_demo = sub.add_parser("demo", help="Run the interactive demo")
    p_demo.add_argument("--steps", type=int, default=50,
                        help="Training steps (default: 50)")
    p_demo.set_defaults(func=cmd_demo)

    p_train = sub.add_parser("train", help="Run the full training pipeline")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
