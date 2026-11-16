"""12주차까지의 최적화·초기화·배치 정규화 비교 및 검증 기반 모델 선택."""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from NeuralNet_fashion import (
    BATCH_SIZE, DEFAULT_DATA_DIR, SEED, describe, evaluate, load_fashion,
    plot_history, prepare_data, train,
)


def main(default_stage="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("optimizer", "init", "batchnorm", "all"), default=default_stage)
    parser.add_argument("--epochs", type=int, default=15, help="기초 비교 에폭")
    parser.add_argument("--long-epochs", type=int, default=35, help="구조 및 무작위 후보 에폭")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--trials", type=int, default=3, help="최종 무작위 학습률 비교 횟수")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_week12")
    args = parser.parse_args()
    if min(args.epochs, args.long_epochs, args.trials) < 1 or args.batch_size < 2:
        parser.error("에폭과 탐색 횟수는 1 이상, 배치 크기는 2 이상이어야 합니다.")
    data = prepare_data(args.data_dir)
    (x, t), training, validation, monitor, _ = data
    if len(training[1]) % args.batch_size == 1:
        parser.error("마지막 배치가 1장이 되지 않는 배치 크기를 선택하세요.")
    base = dict(hidden=[256, 128, 64], optimizer="adam", lr=0.001, init="he",
                batchnorm=True, schedule="constant", l2=0.0, dropout=0.0)
    records = []
    best = None
    best_history = None

    def experiment(name, config, epochs):
        nonlocal best, best_history
        history, epoch, accuracy, loss = train(
            name, config, epochs, *training, monitor, validation=validation,
            batch_size=args.batch_size,
        )
        record = dict(name=name, config=config, epochs=epochs, best_epoch=epoch,
                      val_accuracy=accuracy, val_cross_entropy=loss)
        records.append(record)
        if best is None or (accuracy, -loss) > (best["val_accuracy"], -best["val_cross_entropy"]):
            best, best_history = record, history
        return record

    if args.stage in ("optimizer", "all"):
        for optimizer, rates in [
            ("sgd", [0.01, 0.05]), ("momentum", [0.01, 0.03]),
            ("adagrad", [0.01, 0.05]), ("adam", [0.0005, 0.001]),
        ]:
            for lr in rates:
                experiment(f"optimizer-{optimizer}-{lr}", {**base, "optimizer": optimizer, "lr": lr},
                           args.epochs)
        base = best["config"]

    if args.stage in ("init", "batchnorm", "all"):
        # 3가지 초기화 × BN 켬/끔을 교차 비교해 초기화 의존성도 확인한다.
        flags = (False, True) if args.stage in ("batchnorm", "all") else (True,)
        for batchnorm in flags:
            for init in (0.01, "xavier", "he"):
                experiment(f"init-{init}-BN-{batchnorm}",
                           {**base, "init": init, "batchnorm": batchnorm}, args.epochs)

    if args.stage == "all":
        base = best["config"]
        for hidden in ([256, 128, 64], [480, 288, 144], [512, 256]):
            experiment(f"structure-{hidden}", {**base, "hidden": hidden}, args.long_epochs)
        search_base = best["config"]
        rng = np.random.default_rng(SEED + 12)
        for index in range(args.trials):
            rate = search_base["lr"] * 10 ** rng.uniform(-0.45, 0.35)
            experiment(f"random-lr-{index + 1}", {**search_base, "lr": float(rate)}, args.long_epochs)

    print("\n===== 12주차 검증 비교 결과 =====")
    for record in sorted(records, key=lambda row: (-row["val_accuracy"], row["val_cross_entropy"])):
        print(f"{record['name']} | val={record['val_accuracy']:.2%} | "
              f"CE={record['val_cross_entropy']:.4f} | epoch={record['best_epoch']}/{record['epochs']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(stage=args.stage, seed=SEED, batch_size=args.batch_size,
                  candidates=records, selected=best)
    if args.stage == "all":
        # 검증으로 모든 선택을 확정한 뒤 60,000장 전체 재학습 및 최종 테스트.
        rng = np.random.default_rng(SEED)
        full_monitor = rng.choice(len(t), min(2000, len(t)), replace=False)
        network, final_history = train(
            "12주차 최종 재학습", best["config"], best["epochs"], x, t, full_monitor,
            stop_epoch=best["best_epoch"], batch_size=args.batch_size,
        )
        with (args.output_dir / "network_Team6_week12.pkl").open("wb") as file:
            pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)
        x_test, t_test = load_fashion(args.data_dir, "t10k")
        test_accuracy, test_loss = evaluate(network, x_test, t_test)
        report.update(test_accuracy=test_accuracy, test_cross_entropy=test_loss)
        print(f"\n선택 설정: {describe(best['config'])}")
        print(f"최종 테스트 정확도={test_accuracy:.2%} | CE={test_loss:.4f}", flush=True)
    else:
        final_history = None
    (args.output_dir / f"{args.stage}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if not args.no_plot:
        plot_history(best_history, best["best_epoch"], len(training[1]), args.batch_size, final_history)


if __name__ == "__main__":
    main()
