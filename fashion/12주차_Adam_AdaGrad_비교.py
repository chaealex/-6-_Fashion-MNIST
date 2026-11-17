"""12주차 Adam·AdaGrad 비교: 같은 데이터와 초기 가중치로 설정별 실험."""

import argparse
import json
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from NeuralNet_fashion import DEFAULT_DATA_DIR, SEED, evaluate, load_fashion, prepare_data, train


def plot_results(report, path):
    conditions = list(dict.fromkeys(row["condition"] for row in report["candidates"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    x = np.arange(len(conditions))
    for offset, optimizer, color in [(-0.18, "adagrad", "#087966"), (0.18, "adam", "#27639c")]:
        values = [
            max(row["val_accuracy"] for row in report["candidates"]
                if row["condition"] == condition and row["optimizer"] == optimizer) * 100
            for condition in conditions
        ]
        axes[0].bar(x + offset, values, 0.35, label=optimizer, color=color)
        selected = report["selected"][optimizer]
        axes[1].plot(np.arange(1, selected["settings"]["epochs"] + 1),
                     np.array(selected["epoch_val_accuracy"]) * 100,
                     "o-", label=f"{optimizer}: {selected['condition']}", color=color)
    axes[0].set(title="Week 12 - best validation accuracy", ylabel="Accuracy (%)")
    axes[0].set_xticks(x, conditions, rotation=35, ha="right")
    axes[1].set(title="Selected candidates - validation curve", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.set_ylim(80, 93)
        ax.axhline(90, color="gray", linestyle=":", linewidth=1)
        ax.legend()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "study"), default="study",
                        help="single은 입력 설정, study는 설정별 비교도 실행")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--init", choices=("he", "xavier", "0.01"), default="he")
    parser.add_argument("--batchnorm", type=int, choices=(0, 1), default=1)
    parser.add_argument("--adagrad-lrs", type=float, nargs="+", default=[0.01, 0.025])
    parser.add_argument("--adam-lrs", type=float, nargs="+", default=[0.0003, 0.001])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results_adam_adagrad_week12")
    args = parser.parse_args()
    if not 1 <= len(args.hidden) <= 5 or min(args.hidden) < 1:
        parser.error("은닉층은 1~5개, 노드 수는 양의 정수여야 합니다.")
    if args.batch_size < 2 or args.epochs < 1:
        parser.error("배치 크기는 2 이상, 에폭은 1 이상이어야 합니다.")
    if any(not np.isfinite(lr) or lr <= 0 for lr in args.adagrad_lrs + args.adam_lrs):
        parser.error("학습률은 유한한 양수여야 합니다.")
    _, training, validation, monitor, _ = prepare_data(args.data_dir)
    base = dict(hidden=args.hidden, batch_size=args.batch_size, epochs=args.epochs,
                init=0.01 if args.init == "0.01" else args.init, batchnorm=bool(args.batchnorm))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, best = [], {}

    def experiment(condition, settings, optimizer, lr):
        if settings["batchnorm"] and len(training[1]) % settings["batch_size"] == 1:
            parser.error("마지막 배치가 1장이 되지 않는 배치 크기를 선택하세요.")
        config = dict(hidden=settings["hidden"], init=settings["init"], batchnorm=settings["batchnorm"],
                      optimizer=optimizer, lr=lr, schedule="constant", l2=0.0, dropout=0.0)
        name = f"{condition}-{optimizer}-{lr}"
        checkpoint = args.output_dir / "candidate.pkl"
        started = perf_counter()
        history, epoch, accuracy, loss = train(
            name, config, settings["epochs"], *training, monitor, validation=validation,
            batch_size=settings["batch_size"], checkpoint_path=checkpoint,
        )
        steps = (len(training[1]) + settings["batch_size"] - 1) // settings["batch_size"]
        epoch_losses = [float(np.mean(history[0][i * steps:(i + 1) * steps]))
                        for i in range(settings["epochs"])]
        record = dict(condition=condition, optimizer=optimizer, lr=lr, settings=settings,
                      best_epoch=epoch, val_accuracy=accuracy, val_cross_entropy=loss,
                      elapsed_seconds=perf_counter() - started, epoch_train_ce=epoch_losses,
                      epoch_train_sample_accuracy=history[2], epoch_val_accuracy=history[3])
        records.append(record)
        if optimizer not in best or (accuracy, -loss) > (
            best[optimizer]["val_accuracy"], -best[optimizer]["val_cross_entropy"]
        ):
            best[optimizer] = record
            shutil.copyfile(checkpoint, args.output_dir / f"{optimizer}_best.pkl")
        (args.output_dir / "validation.json").write_text(
            json.dumps(dict(seed=SEED, candidates=records, selected=best), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 기준 설정에서 방법별 학습률을 고르고, 이후 조건은 한 변수씩 변경한다.
    for optimizer, rates in (("adagrad", args.adagrad_lrs), ("adam", args.adam_lrs)):
        for lr in rates:
            experiment("baseline", base, optimizer, lr)
    rates = {optimizer: record["lr"] for optimizer, record in best.items()}
    if args.mode == "study":
        conditions = [
            ("batch64", {**base, "batch_size": 64}),
            ("batch256", {**base, "batch_size": 256}),
            ("small", {**base, "hidden": [128, 64]}),
            ("wide", {**base, "hidden": [512, 256]}),
            ("long", {**base, "epochs": args.epochs * 2}),
            ("xavier", {**base, "init": "xavier"}),
            ("without_bn", {**base, "batchnorm": False}),
        ]
        for name, settings in conditions:
            for optimizer in ("adagrad", "adam"):
                experiment(name, settings, optimizer, rates[optimizer])

    # 방법별 모델을 검증 점수로 고른 뒤 각각 한 번씩 최종 평가한다.
    x_test, t_test = load_fashion(args.data_dir, "t10k")
    tests = {}
    for optimizer in ("adagrad", "adam"):
        with (args.output_dir / f"{optimizer}_best.pkl").open("rb") as file:
            network = pickle.load(file)
        accuracy, loss = evaluate(network, x_test, t_test)
        tests[optimizer] = dict(accuracy=accuracy, cross_entropy=loss)
        print(f"FINAL {optimizer}: test={accuracy:.2%}, CE={loss:.4f}", flush=True)
    report = dict(seed=SEED, mode=args.mode, candidates=records, selected=best, test=tests,
                  evaluation="best_validation_checkpoint_54000")
    (args.output_dir / "final.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    plot_results(report, args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
