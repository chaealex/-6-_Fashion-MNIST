"""Fashion-MNIST: Adam vs AdaGrad 공정 비교 (6조 팀플 추가 실험).

실행 위치: C:\\Neural_Net\\team_github
    python fashion/optimizer_compare.py                # 빠른 작동 검사 (2 epoch)
    python fashion/optimizer_compare.py --full         # 정식 비교 (10 epoch)

기존 신경망 모델과 common 코드는 수정하지 않습니다.
데이터는 훈련용 60,000장만 읽고, 테스트 파일은 사용하지 않습니다.
"""
import argparse
import csv
import gzip
import hashlib
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 결과 그래프 저장
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import Adam, AdaGrad

SEED = 42
HIDDEN = [480, 288, 144]  # 기존 최종 후보 구조. 입력 784, 출력 10
OFFICIAL_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
}


def load_training_data():
    """Fashion-MNIST 공식 훈련 원본인지 검사하고 0~1로 정규화한다."""
    folder = ROOT / "Fashion_mnist" / "data" / "fashion"
    for filename, expected in OFFICIAL_MD5.items():
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{filename}는 공식 Fashion-MNIST 원본과 다릅니다. 손글씨 MNIST인지 확인하세요.")
    with gzip.open(folder / "train-images-idx3-ubyte.gz", "rb") as stream:
        images = np.frombuffer(stream.read(), dtype=np.uint8, offset=16).copy()
    with gzip.open(folder / "train-labels-idx1-ubyte.gz", "rb") as stream:
        labels = np.frombuffer(stream.read(), dtype=np.uint8, offset=8).copy()
    if images.size != 60000 * 784 or labels.size != 60000:
        raise ValueError("Fashion-MNIST 훈련 파일의 크기가 올바르지 않습니다.")
    x = images.reshape(-1, 784).astype(np.float32) / 255.0
    return x, labels


def split_training(x, t, quick=False):
    """원본 훈련 60,000장만 학습/검증(54,000/6,000)으로 분리한다."""
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(t == label))
        val_idx.extend(indices[:600])
        train_idx.extend(indices[600:])
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    if quick:
        # 프로그램 작동 여부를 먼저 확인하는 소규모 시험 실행
        train_idx = train_idx[:5000]
        val_idx = val_idx[:1000]
    return x[train_idx], t[train_idx], x[val_idx], t[val_idx]


def make_network():
    """두 최적화 함수가 완전히 같은 초기 가중치에서 출발하도록 만든다."""
    np.random.seed(SEED)
    net = MultiLayerNetExtend(
        input_size=784, hidden_size_list=HIDDEN, output_size=10,
        activation="relu", weight_init_std="he", use_batchnorm=True,
        use_dropout=False, weight_decay_lambda=0.0,
    )
    # 교재 Affine/BatchNorm 계층과 params 딕셔너리의 배열 참조 일치
    for name in net.params:
        net.params[name] = net.params[name].astype(np.float32)
    for i in range(1, len(HIDDEN) + 2):
        layer = net.layers[f"Affine{i}"]
        layer.W = net.params[f"W{i}"]
        layer.b = net.params[f"b{i}"]
    for i, width in enumerate(HIDDEN, start=1):
        layer = net.layers[f"BatchNorm{i}"]
        layer.gamma = net.params[f"gamma{i}"]
        layer.beta = net.params[f"beta{i}"]
        layer.running_mean = np.zeros(width, dtype=np.float32)
        layer.running_var = np.zeros(width, dtype=np.float32)
    return net


def metrics(net, x, labels, batch_size=512):
    """추론 모드로 정확도와 교차 엔트로피 계산 (BatchNorm 통계 변경 안 함)."""
    correct, total_ce = 0, 0.0
    for start in range(0, len(labels), batch_size):
        t_batch = labels[start:start + batch_size]
        logits = net.predict(x[start:start + batch_size], train_flg=False)
        if not np.isfinite(logits).all():
            raise FloatingPointError("신경망 출력에 NaN 또는 Inf가 발생했습니다.")
        correct += np.count_nonzero(logits.argmax(axis=1) == t_batch)
        shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
        total_ce += (np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(t_batch)), t_batch]).sum()
    return float(correct / len(labels)), float(total_ce / len(labels))


def run_one(name, lr, x_train, t_train, x_val, t_val, epochs, batch_size):
    net = make_network()
    opt = Adam(lr=lr) if name == "Adam" else AdaGrad(lr=lr)
    rng = np.random.default_rng(SEED + 1)  # 두 방법의 미니배치 순서도 동일
    monitor_x, monitor_t = x_train[:min(2000, len(x_train))], t_train[:min(2000, len(x_train))]
    history = []
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            gradients = net.gradient(x_train[idx], t_train[idx])
            opt.update(net.params, gradients)
        train_acc, train_loss = metrics(net, monitor_x, monitor_t)
        val_acc, val_loss = metrics(net, x_val, t_val)
        row = {
            "optimizer": name, "learning_rate": lr, "batch_size": batch_size,
            "epoch": epoch, "train_sample_accuracy": train_acc,
            "train_sample_ce": train_loss, "val_accuracy": val_acc,
            "val_ce": val_loss, "epoch_seconds": round(perf_counter() - started, 2),
        }
        history.append(row)
        print(f"[{name}] {epoch}/{epochs} epoch | train(sample)={train_acc:.2%} "
              f"| validation={val_acc:.2%} | val loss={val_loss:.4f}", flush=True)
    return history


def save_outputs(histories, output_dir, mode):
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [r for history in histories.values() for r in history]
    with (output_dir / "optimizer_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for field, title, ylabel, filename in [
        ("val_accuracy", "Validation accuracy", "Accuracy", "validation_accuracy.png"),
        ("val_ce", "Validation cross-entropy", "Loss", "validation_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, history in histories.items():
            ax.plot([r["epoch"] for r in history], [r[field] for r in history], marker="o", label=name)
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"Fashion-MNIST ({mode}) - {title}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
    print("\n최종 에폭 결과 (학습/검증 비교 실험, 테스트 정확도 아님):")
    for name, history in histories.items():
        best = max(history, key=lambda r: (r["val_accuracy"], -r["val_ce"]))
        print(f"{name}: 최고 검증 정확도={best['val_accuracy']:.2%} (epoch {best['epoch']}), "
              f"그때 검증 손실={best['val_ce']:.4f}")
    print(f"CSV와 그래프 저장 완료: {output_dir}")
    if mode == "quick":
        print("빠른 동작 검사 결과는 공식 전체 학습 성능으로 사용하지 마세요.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="54,000장 학습/6,000장 검증, 기본 10에폭")
    parser.add_argument("--epochs", type=int, default=None, help="에폭 수를 직접 지정")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--adam-lr", type=float, default=0.001)
    parser.add_argument("--adagrad-lr", type=float, default=0.01)
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (10 if args.full else 2)
    if epochs < 1 or args.batch_size < 2 or args.adam_lr <= 0 or args.adagrad_lr <= 0:
        parser.error("에폭은 1 이상, 배치는 2 이상, 학습률은 양수여야 합니다.")
    mode = "full" if args.full else "quick"
    x, t = load_training_data()
    a, b, c, d = split_training(x, t, quick=not args.full)
    print(f"Fashion-MNIST | {mode} | 학습 {len(b):,}장, 검증 {len(d):,}장")
    print("구조: 784-480-288-144-10, BatchNorm ON, Dropout OFF, L2 OFF")
    print("테스트 데이터는 이번 실험에 사용하지 않습니다.\n", flush=True)
    histories = {}
    for name, lr in [("Adam", args.adam_lr), ("AdaGrad", args.adagrad_lr)]:
        histories[name] = run_one(name, lr, a, b, c, d, epochs, args.batch_size)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_outputs(histories, HERE / "optimizer_results" / f"{mode}_{stamp}", mode)


if __name__ == "__main__":
    main()
