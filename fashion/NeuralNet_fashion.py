"""Fashion-MNIST 팀 과제: 수동 비교 → 무작위 탐색 → 최종 평가."""

import argparse
import gzip
from pathlib import Path
import pickle
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import Adam


SEED = 42
BATCH_SIZE = 256
LOG_INTERVAL = 50
TARGET_ACCURACY = 0.90


def load_fashion(data_dir, kind):
    # 공식 mnist_reader.py 방식: IDX 헤더를 건너뛰고 픽셀과 레이블을 읽는다.
    # https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    with gzip.open(data_dir / f"{kind}-labels-idx1-ubyte.gz", "rb") as file:
        labels = np.frombuffer(file.read(), dtype=np.uint8, offset=8)
    with gzip.open(data_dir / f"{kind}-images-idx3-ubyte.gz", "rb") as file:
        images = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
    return images.reshape(len(labels), 784).astype(np.float32) / 255.0, labels


def evaluate(network, x, t):
    """전체 데이터의 정확도와 교차 엔트로피를 메모리에 맞게 나누어 계산."""
    correct = 0
    loss = 0.0
    for start in range(0, len(t), 1024):
        target = t[start:start + 1024]
        scores = network.predict(x[start:start + 1024])
        correct += np.count_nonzero(scores.argmax(axis=1) == target)
        loss += network.last_layer.forward(scores, target) * len(target)
    return correct / len(t), loss / len(t)


def describe(config):
    shape = " → ".join(map(str, [784, *config["hidden"], 10]))
    return (
        f"{shape} ({len(config['hidden']) + 1}층), ReLU, Adam, "
        f"lr={config['lr']:.5g}, L2={config['l2']:.5g}, dropout={config['dropout']:.2f}"
    )


def train(name, config, epochs, x_train, t_train, monitor, validation=None, stop_epoch=None):
    # 각 후보가 같은 난수 시드와 배치 순서를 사용하도록 맞춘다.
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=config["hidden"],
        output_size=10,
        activation="relu",
        weight_init_std="he",
        weight_decay_lambda=config["l2"],
        use_dropout=config["dropout"] > 0,
        dropout_ration=config["dropout"],
        use_batchnorm=True,
    )
    # 교재 계층에서 가중치를 참조하므로 양쪽을 같은 float32 배열에 연결한다.
    for key, value in network.params.items():
        network.params[key] = value.astype(np.float32)
    for index, size in enumerate(config["hidden"], 1):
        bn = network.layers[f"BatchNorm{index}"]
        bn.gamma, bn.beta = network.params[f"gamma{index}"], network.params[f"beta{index}"]
        bn.running_mean = np.zeros(size, dtype=np.float32)
        bn.running_var = np.zeros(size, dtype=np.float32)
    for index in range(1, len(config["hidden"]) + 2):
        layer = network.layers[f"Affine{index}"]
        layer.W, layer.b = network.params[f"W{index}"], network.params[f"b{index}"]
    optimizer = Adam(lr=config["lr"])

    steps_per_epoch = (len(t_train) + BATCH_SIZE - 1) // BATCH_SIZE
    run_epochs = stop_epoch if stop_epoch is not None else epochs
    total_steps = steps_per_epoch * run_epochs
    batch_losses, log_steps, train_scores, val_scores = [], [], [], []
    epoch_scores = []
    best_key = (-1.0, float("-inf"))
    monitor_x, monitor_t = x_train[monitor], t_train[monitor]
    step = 0
    print(f"\n[{name}] {describe(config)} | {run_epochs} epochs", flush=True)
    for epoch in range(1, run_epochs + 1):
        # 모든 후보에 같은 학습률 감소 규칙을 적용한다.
        progress = (epoch - 1) / epochs
        optimizer.lr = config["lr"] * (0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2)
        order = rng.permutation(len(t_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            # gradient()가 순전파와 역전파를 수행한다. 갱신 직전의 배치 손실을 기록한다.
            gradients = network.gradient(x_train[batch], t_train[batch])
            batch_losses.append(network.last_layer.loss)
            optimizer.update(network.params, gradients)
            step += 1
            log_now = step % LOG_INTERVAL == 0 or step == total_steps
            epoch_end = step % steps_per_epoch == 0
            if validation is not None and (log_now or epoch_end):
                val_acc, val_loss = evaluate(network, *validation)
                if epoch_end:
                    epoch_scores.append(val_acc)
                    key = (val_acc, -val_loss)
                    if key > best_key:
                        best_key, best_epoch = key, epoch
            if log_now:
                train_acc = network.accuracy(monitor_x, monitor_t)
                log_steps.append(step)
                train_scores.append(train_acc)
                recent_loss = np.mean(batch_losses[-LOG_INTERVAL:])
                validation_log = ""
                if validation is not None:
                    val_scores.append(val_acc)
                    validation_log = f" | val={val_acc:.2%} | val loss={val_loss:.4f}"
                print(
                    f"batch {step:>5}/{total_steps} | epoch {epoch:>2}/{run_epochs} | "
                    f"loss={recent_loss:.4f} | train(표본)={train_acc:.2%}{validation_log}",
                    flush=True,
                )
    history = (batch_losses, log_steps, train_scores, val_scores)
    if validation is None:
        return network, history
    checkpoints = [i for i in (5, 10, 15, 25, 35, 45) if i < epochs] + [epochs]
    checkpoint_log = ", ".join(f"{i} epoch={epoch_scores[i - 1]:.2%}" for i in checkpoints)
    print(f"[{name}] 에폭별 검증: {checkpoint_log} | 최고 {best_epoch} epoch={best_key[0]:.2%}", flush=True)
    return history, best_epoch, best_key[0], -best_key[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=HERE.parent / "Fashion_mnist" / "data" / "fashion"
    )
    parser.add_argument("--epochs", type=int, default=15, help="수동 탐색의 후보당 에폭 수")
    parser.add_argument("--search-epochs", type=int, default=45, help="장기 학습과 랜덤 탐색의 후보당 에폭 수")
    parser.add_argument("--trials", type=int, default=3, help="무작위 탐색 횟수")
    parser.add_argument("--save-model", action="store_true", help="선택 모델을 network_Team6.pkl로 저장")
    args = parser.parse_args()
    if min(args.epochs, args.search_epochs, args.trials) < 1:
        parser.error("에폭 수와 무작위 탐색 횟수는 1 이상이어야 합니다.")

    started = perf_counter()
    x, t = load_fashion(args.data_dir, "train")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(t))
    split = len(order) // 10
    x_val, t_val = x[order[:split]], t[order[:split]]
    x_train, t_train = x[order[split:]], t[order[split:]]
    # 50배치마다 고정된 학습 표본 2,000장과 검증 전체를 평가한다.
    monitor = rng.choice(len(t_train), min(2000, len(t_train)), replace=False)
    print(f"Fashion-MNIST | 학습 {len(t_train):,}장 | 검증 {len(t_val):,}장")

    # 입력층을 제외하고 은닉층 최대 5개와 출력층 1개를 사용한다.
    baseline = dict(lr=0.001, l2=0.00002, dropout=0.15)
    widths = ([256], [128, 64], [384, 192], [320, 192, 96], [256, 192, 128, 96, 64])
    manual = [{**baseline, "hidden": hidden} for hidden in widths]
    results = []
    best_key = (-1.0, float("-inf"))
    for index in range(len(manual) + 1 + args.trials):
        epochs = args.epochs if index < len(manual) else args.search_epochs
        if index < len(manual):
            name = f"수동 {index + 1}"
            config = manual[index]
        elif index == len(manual):
            name, search_base = "장기 학습", best_config
            config = search_base
        else:
            name = f"랜덤 {index - len(manual)}"
            # 검증 결과가 좋은 깊이를 유지하며 폭과 정규화 강도를 무작위로 바꾼다.
            scale = rng.choice([1.0, 1.25, 1.5])
            config = {
                **search_base,
                "hidden": [round(size * scale) for size in search_base["hidden"]],
                "lr": search_base["lr"] * 10 ** rng.uniform(-0.2, 0.25),
                "l2": 10 ** rng.uniform(-5.5, -4.0),
                "dropout": rng.uniform(0.10, 0.23),
            }

        history, best_epoch, val_acc, val_loss = train(
            name, config, epochs, x_train, t_train, monitor, validation=(x_val, t_val)
        )
        results.append((name, config, val_acc, val_loss, best_epoch, epochs))
        key = (val_acc, -val_loss)
        if key > best_key:
            best_key = key
            best_name, best_config = name, config
            selected_epoch, selected_horizon, best_history = best_epoch, epochs, history

    print("\n===== 후보별 최고 검증 결과 =====")
    for name, config, val_acc, val_loss, epoch, horizon in sorted(results, key=lambda row: (-row[2], row[3])):
        print(f"{name} | epoch={epoch}/{horizon} | 검증={val_acc:.2%} | loss={val_loss:.4f} | {describe(config)}")

    # 검증으로 구조와 종료 에폭을 확정한 뒤, 원래 학습셋 전체로 한 번 더 학습한다.
    full_monitor = rng.choice(len(t), min(2000, len(t)), replace=False)
    best_network, final_history = train(
        "전체 학습셋 재학습", best_config, selected_horizon, x, t, full_monitor, stop_epoch=selected_epoch
    )
    # 계층에 테스트 순전파 중간값이 담기기 전에 제출용 모델을 저장한다.
    if args.save_model:
        path = HERE / "network_Team6.pkl"
        with path.open("wb") as file:
            pickle.dump(best_network, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"모델 저장: {path}")

    x_test, t_test = load_fashion(args.data_dir, "t10k")
    print(f"\n===== 최종 선택: {best_name} =====")
    print(describe(best_config))
    train_acc, train_loss = evaluate(best_network, x, t)
    test_acc, test_loss = evaluate(best_network, x_test, t_test)
    print(f"선택 에폭={selected_epoch} | 선택 당시 검증 정확도={best_key[0]:.2%}")
    print(f"최종 학습 ({len(t):,}장) | 정확도={train_acc:.2%} | 손실={train_loss:.4f}")
    print(f"최종 테스트 ({len(t_test):,}장) | 정확도={test_acc:.2%} | 손실={test_loss:.4f}")
    print(f"테스트 90% 목표: {'달성' if test_acc >= TARGET_ACCURACY else '미달'}")
    print(f"총 실행 시간: {perf_counter() - started:.1f}초", flush=True)

    losses = final_history[0]
    _, steps, train_scores, val_scores = best_history
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4))
    loss_ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=0.8)
    loss_ax.set(title="Final Training Loss", xlabel="Batch update", ylabel="Cross-entropy")
    acc_ax.plot(steps, train_scores, label="Train (sample)")
    acc_ax.plot(steps, val_scores, "--", label="Validation")
    acc_ax.axvline(selected_epoch * ((len(t_train) + BATCH_SIZE - 1) // BATCH_SIZE),
                   color="gray", linestyle=":", label="Selected epoch")
    acc_ax.set(title="Model Selection Accuracy", xlabel="Batch update", ylabel="Accuracy", ylim=(0, 1))
    acc_ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
