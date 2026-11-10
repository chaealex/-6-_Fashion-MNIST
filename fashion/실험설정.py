"""CMD에서 설정을 입력하고 Fashion-MNIST 실험 하나를 실행."""

import argparse
import math
from pathlib import Path

from NeuralNet_fashion import BATCH_SIZE, DEFAULT_DATA_DIR, describe, run_experiment


def ask(label, default, parse, valid, hint):
    while True:
        entered = input(f"{label} [{default}]: ").strip()
        try:
            value = parse(entered or str(default))
        except ValueError:
            print(hint)
            continue
        if valid(value):
            return value
        print(hint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("===== Fashion-MNIST 실험 설정 =====")
    print("Enter는 기본값 사용. 입력층 784개와 출력층 10개는 고정.")
    try:
        depth = ask("은닉층 수 (1~5)", 3, int, lambda n: 1 <= n <= 5,
                    "은닉층 수는 1~5 사이의 정수로 입력하세요.")
        default_widths = " ".join(map(str, [480, 288, 144, 96, 64][:depth]))
        widths = ask(
            "층별 노드 수 (공백 또는 쉼표로 구분)", default_widths,
            lambda text: [int(value) for value in text.replace(",", " ").split()],
            lambda values: len(values) == depth and all(value > 0 for value in values),
            f"1 이상의 정수를 정확히 {depth}개 입력하세요.",
        )
        l2 = ask("L2 계수 (예: 8.4e-6)", "8.4e-6", float,
                 lambda n: math.isfinite(n) and n >= 0, "L2는 유한한 0 이상의 수를 입력하세요.")
        dropout = ask("드롭아웃 비율 (예: 0.2 = 20%)", 0.2, float, lambda n: 0 <= n < 1,
                      "드롭아웃은 0 이상 1 미만의 소수로 입력하세요.")
        batch_size = ask("배치 크기", BATCH_SIZE, int, lambda n: n > 0, "1 이상의 정수를 입력하세요.")
        epochs = ask("에폭", 15, int, lambda n: n > 0, "1 이상의 정수를 입력하세요.")
        lr = ask("초기 학습률", 0.00127, float, lambda n: math.isfinite(n) and n > 0,
                 "유한한 양수를 입력하세요.")
        config = dict(hidden=widths, l2=l2, dropout=dropout, lr=lr)
        print(f"\n선택 설정: {describe(config)} | 배치={batch_size} | 에폭={epochs}")
        run_experiment(config, epochs=epochs, batch_size=batch_size, data_dir=args.data_dir,
                       show_plot=not args.no_plot)
    except (EOFError, KeyboardInterrupt):
        print("\n실험을 종료합니다.")


if __name__ == "__main__":
    main()
