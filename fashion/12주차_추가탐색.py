"""배치 크기·학습률·노드 수 추가 비교. 최고 검증 체크포인트를 최종 테스트."""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from NeuralNet_fashion import DEFAULT_DATA_DIR, SEED, evaluate, load_fashion, prepare_data, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_week12_extra")
    args = parser.parse_args()
    if args.epochs < 1 or args.trials < 1:
        parser.error("에폭과 무작위 탐색 횟수는 1 이상이어야 합니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, training, validation, monitor, _ = prepare_data(args.data_dir)
    base = dict(optimizer="adagrad", lr=0.025, hidden=[256, 128, 64], init="he",
                batchnorm=True, schedule="constant", l2=0.0, dropout=0.0)
    # 같은 구조에서 배치 크기를 먼저 비교하고, 학습률과 모델 폭을 분리해 변경.
    candidates = [
        ({**base}, 64),
        ({**base}, 128),
        ({**base}, 512),
        ({**base, "lr": 0.01}, 128),
        ({**base, "lr": 0.05}, 128),
        ({**base, "hidden": [128, 64]}, 128),
        ({**base, "hidden": [512, 256]}, 128),
        ({**base, "optimizer": "adam", "lr": 0.0003}, 128),
    ]
    records = []
    best = None
    rng = np.random.default_rng(SEED + 120)
    for index in range(len(candidates) + args.trials):
        if index < len(candidates):
            config, batch = candidates[index]
            name = f"manual-{index + 1}"
        else:
            # 수동 비교 최고 후보 주변에서 세 변수만 무작위 탐색.
            batch = int(rng.choice([64, 128, 256, 512]))
            scale = float(rng.choice([0.75, 1.0, 1.25]))
            config = {**search_base,
                      "hidden": [max(16, round(width * scale)) for width in search_base["hidden"]],
                      "lr": float(search_base["lr"] * 10 ** rng.uniform(-0.3, 0.25))}
            name = f"random-{index - len(candidates) + 1}"
        checkpoint = args.output_dir / "candidate.pkl"
        _, epoch, accuracy, loss = train(
            name, config, args.epochs, *training, monitor, validation=validation,
            batch_size=batch, checkpoint_path=checkpoint,
        )
        record = dict(name=name, config=config, batch_size=batch, epochs=args.epochs,
                      best_epoch=epoch, val_accuracy=accuracy, val_cross_entropy=loss)
        records.append(record)
        if best is None or (accuracy, -loss) > (best["val_accuracy"], -best["val_cross_entropy"]):
            best = record
            shutil.copyfile(checkpoint, args.output_dir / "network_Team6_week12_extra.pkl")
        if index == len(candidates) - 1:
            search_base = best["config"]
        (args.output_dir / "validation.json").write_text(
            json.dumps(dict(seed=SEED, candidates=records, selected=best), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 검증 최고 시점에 저장한 모델 자체를 최종 평가한다.
    with (args.output_dir / "network_Team6_week12_extra.pkl").open("rb") as file:
        network = pickle.load(file)
    x_test, t_test = load_fashion(args.data_dir, "t10k")
    test_accuracy, test_loss = evaluate(network, x_test, t_test)
    report = dict(seed=SEED, candidates=records, selected=best,
                  final_method="best_validation_checkpoint_54000",
                  test_accuracy=test_accuracy, test_cross_entropy=test_loss)
    (args.output_dir / "final.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\n===== 추가 탐색 최종 결과 =====")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"테스트 정확도={test_accuracy:.2%} | CE={test_loss:.4f} | "
          f"90% 목표={'달성' if test_accuracy > 0.90 else '미달'}", flush=True)


if __name__ == "__main__":
    main()
