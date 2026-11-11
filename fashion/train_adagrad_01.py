"""6조 Fashion-MNIST: AdaGrad를 이용해 성능 높은 단일 MLP 탐색.

팀 과제 흐름: 수동 비교 -> Randomized Search -> 상위 후보 장기 학습 -> 단일 모델 저장.
실행 위치: C:\\Neural_Net\\team_github
    python fashion/train_adagrad_best.py --quick    # 파일·학습·저장 기능 점검
    python fashion/train_adagrad_best.py --full     # 수동 탐색/무작위 탐색/최종 후보

작업 중 Ctrl+C로 중단했다면 출력된 run 폴더로 재개:
    python fashion/train_adagrad_best.py --resume "결과_폴더_경로"

최종 설정이 완전히 결정된 후에만 (한 번) 테스트:
    python fashion/train_adagrad_best.py --evaluate "결과_폴더_경로"

교재 4~6장 기법만 사용: 완전연결 신경망, 역전파, AdaGrad, ReLU/He,
BatchNorm, L2/Dropout, 검증 데이터, 수동+무작위 탐색.
입력층을 포함해 최대 6층 (은닉층 최대 4개), CNN/앙상블 없음.
원본 및 팀원 파일은 수정하지 않는다. 테스트 이미지는 탐색 중 읽지 않는다.
"""

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import AdaGrad

SEED = 42
# 공식 Fashion-MNIST 원본 MD5. 일반 MNIST 파일을 잘못 넣는 것을 방지함.
OFFICIAL = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def data_directory():
    return ROOT / "Fashion_mnist" / "data" / "fashion"


def load_data(kind):
    """train 또는 t10k만 읽으며 원본 체크섬과 크기를 검사한다."""
    if kind not in ("train", "t10k"):
        raise ValueError("kind는 train 또는 t10k만 가능합니다.")
    folder = data_directory()
    for filename in (f"{kind}-images-idx3-ubyte.gz", f"{kind}-labels-idx1-ubyte.gz"):
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if md5 != OFFICIAL[filename]:
            raise ValueError(f"Fashion-MNIST 공식 원본과 다른 파일입니다: {filename}")

    with gzip.open(folder / f"{kind}-images-idx3-ubyte.gz", "rb") as f:
        pixels = np.frombuffer(f.read(), dtype=np.uint8, offset=16).copy()
    with gzip.open(folder / f"{kind}-labels-idx1-ubyte.gz", "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8).copy()
    count = 60000 if kind == "train" else 10000
    if pixels.size != count * 784 or labels.size != count:
        raise ValueError("데이터 개수가 올바르지 않습니다.")
    x = pixels.reshape(count, 784).astype(np.float32) / 255.0
    return x, labels


def train_validation_split(x, labels, quick=False):
    """전체 훈련 원본 60,000장을 층화해 54,000/6,000으로 분리."""
    rng = np.random.default_rng(SEED)
    tr, va = [], []
    for label in range(10):
        ids = rng.permutation(np.where(labels == label)[0])
        va.extend(ids[:600])
        tr.extend(ids[600:])
    tr, va = rng.permutation(tr), rng.permutation(va)
    if quick:
        tr, va = tr[:2000], va[:400]  # 성능 판단 금지: 기능 테스트용
    return x[tr], labels[tr], x[va], labels[va]


def validate_config(config):
    widths = config["hidden"]
    # 입력층까지 센 경우에도 1 + 은닉층 4 + 출력층 1 = 6층
    if not (1 <= len(widths) <= 4) or any(w < 1 for w in widths):
        raise ValueError("과제 조건: 은닉층 최대 4개, 각 층 노드 1개 이상")
    if config["lr"] <= 0 or not 0 <= config["dropout"] < 1 or config["l2"] < 0:
        raise ValueError("학습률/정규화 설정이 잘못되었습니다.")
    if config["batch"] < 2:
        raise ValueError("배치 크기는 2 이상이어야 합니다.")


def create_model(config, seed=SEED):
    validate_config(config)
    np.random.seed(seed)
    net = MultiLayerNetExtend(
        input_size=784, hidden_size_list=config["hidden"], output_size=10,
        activation="relu", weight_init_std="he", use_batchnorm=True,
        use_dropout=config["dropout"] > 0,
        dropout_ration=config["dropout"],
        weight_decay_lambda=config["l2"],
    )
    # 공통 모듈은 건드리지 않는다. float32 변환 후 layer/params 연결만 일치시킨다.
    for name in net.params:
        net.params[name] = net.params[name].astype(np.float32)
    for i in range(1, len(config["hidden"]) + 2):
        aff = net.layers[f"Affine{i}"]
        aff.W, aff.b = net.params[f"W{i}"], net.params[f"b{i}"]
    for i, width in enumerate(config["hidden"], 1):
        bn = net.layers[f"BatchNorm{i}"]
        bn.gamma, bn.beta = net.params[f"gamma{i}"], net.params[f"beta{i}"]
        bn.running_mean = np.zeros(width, dtype=np.float32)
        bn.running_var = np.zeros(width, dtype=np.float32)
    return net


def accuracy_loss(net, x, labels, batch=512):
    """추론 모드 검증. 배치정규화 이동통계 갱신이나 역전파는 없음."""
    correct, ce_sum = 0, 0.0
    for start in range(0, len(labels), batch):
        t = labels[start:start + batch]
        outputs = net.predict(x[start:start + batch], train_flg=False)
        if not np.isfinite(outputs).all():
            raise FloatingPointError("추론에서 NaN/Inf가 발생했습니다.")
        correct += int(np.sum(np.argmax(outputs, axis=1) == t))
        shifted = outputs.astype(np.float64) - outputs.max(axis=1, keepdims=True)
        ce_sum += float((np.log(np.exp(shifted).sum(axis=1)) -
                         shifted[np.arange(len(t)), t]).sum())
    return correct / len(labels), ce_sum / len(labels)


def save_checkpoint(net, config, filepath):
    """학습 중 최고 상태의 가중치/BN 통계를 clean 모델에 복사해 저장."""
    state = {key: value.copy() for key, value in net.params.items()}
    bn_stats = {
        key: (layer.running_mean.copy(), layer.running_var.copy())
        for key, layer in net.layers.items() if key.startswith("BatchNorm")
    }
    # 새 클래스가 아니라 같은 교재의 MultiLayerNetExtend 객체를 저장한다.
    clean = create_model(config)
    for key, value in state.items():
        clean.params[key][...] = value
    for key, (mean, var) in bn_stats.items():
        clean.layers[key].running_mean = mean
        clean.layers[key].running_var = var
    tmp = filepath.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(clean, f, protocol=4)
    tmp.replace(filepath)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_trial(name, config, epochs, dataset, trial_dir, quick=False):
    """검증 CE 기준 조기 종료, 최고 체크포인트 보존; 시험셋 사용하지 않음."""
    validate_config(config)
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_path = trial_dir / "result.json"
    if result_path.exists() and (trial_dir / "best_model.pkl").exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["config"] != config or result["max_epochs"] != epochs:
            raise ValueError(f"기존 실험 설정과 다릅니다: {trial_dir}")
        print(f"[{name}] 이미 완료되어 기록을 불러옵니다.", flush=True)
        return result

    x_train, t_train, x_val, t_val = dataset
    net = create_model(config)
    opt = AdaGrad(lr=config["lr"])
    rng = np.random.default_rng(SEED + 1)
    sample_x, sample_t = x_train[:min(len(t_train), 2000)], t_train[:min(len(t_train), 2000)]
    history = []
    best_ce = float("inf")
    best_epoch, bad = 0, 0
    best_val_acc = 0.0
    started = perf_counter()
    patience = 4 if quick else 7
    print(f"\n[{name}] hidden={config['hidden']}, AdaGrad lr={config['lr']:.5g}, "
          f"dropout={config['dropout']:.2f}, L2={config['l2']:.5g}, "
          f"batch={config['batch']}, epochs={epochs}", flush=True)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(t_train))
        for start in range(0, len(order), config["batch"]):
            idx = order[start:start + config["batch"]]
            grads = net.gradient(x_train[idx], t_train[idx])
            if any(not np.all(np.isfinite(g)) for g in grads.values()):
                raise FloatingPointError(f"[{name}] 학습 중 기울기 NaN/Inf")
            opt.update(net.params, grads)

        tr_acc, tr_ce = accuracy_loss(net, sample_x, sample_t)
        val_acc, val_ce = accuracy_loss(net, x_val, t_val)
        row = {"epoch": epoch, "train_sample_accuracy": round(tr_acc, 6),
               "train_sample_ce": round(tr_ce, 6),
               "val_accuracy": round(val_acc, 6), "val_ce": round(val_ce, 6)}
        history.append(row)
        write_csv(trial_dir / "history.csv", history)
        print(f"[{name}] {epoch}/{epochs} epoch | train(sample)={tr_acc:.2%} "
              f"| val={val_acc:.2%} | val CE={val_ce:.4f}", flush=True)

        # 정확도는 소수의 데이터에서 요동칠 수 있으므로 검증 손실 최소 상태 저장.
        if val_ce < best_ce - 1e-4:
            best_ce, best_epoch, best_val_acc, bad = val_ce, epoch, val_acc, 0
            # 저장용 모델 생성이 전역 난수 상태를 초기화하지 않도록 보호한다.
            rng_state = np.random.get_state()
            try:
                save_checkpoint(net, config, trial_dir / "best_model.pkl")
            finally:
                np.random.set_state(rng_state)
        else:
            bad += 1
        if epoch >= (2 if quick else 10) and bad >= patience:
            print(f"[{name}] 검증 CE 정체 -> {epoch} epoch에서 조기 종료", flush=True)
            break

    # 복원한 모델이 보관한 검증값을 재현하는지 검사한다.
    with (trial_dir / "best_model.pkl").open("rb") as f:
        selected = pickle.load(f)  # 방금 로컬에서 만든 체크포인트만 로드
    restored_acc, restored_ce = accuracy_loss(selected, x_val, t_val)
    if abs(restored_acc - best_val_acc) > 1e-9 or abs(restored_ce - best_ce) > 1e-5:
        raise RuntimeError("학습 중 최고 체크포인트와 복원 모델의 검증값이 다릅니다.")
    result = {"name": name, "config": config, "max_epochs": epochs,
              "best_epoch": best_epoch, "val_accuracy": round(restored_acc, 6),
              "val_ce": round(restored_ce, 6), "max_observed_val_accuracy": round(max(h["val_accuracy"] for h in history), 6),
              "epochs_completed": len(history), "elapsed_seconds": round(perf_counter() - started, 2)}
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def manual_configs(quick=False):
    if quick:
        return [
            ("manual_smoke", {"hidden": [32, 16], "lr": .01, "l2": 0.0,
                              "dropout": 0.0, "batch": 128}),
        ]
    # 기존 팀 후보(480-288-144)에서 시작해 구조/규제 강도를 수동으로 조정한다.
    return [
        ("manual_1_no_regularization", {"hidden": [480, 288, 144], "lr": .01, "l2": 0.0, "dropout": 0.0, "batch": 128}),
        ("manual_2_mild_regularization", {"hidden": [480, 288, 144], "lr": .008, "l2": 0.00005, "dropout": 0.10, "batch": 128}),
        ("manual_3_more_regularization", {"hidden": [480, 288, 144], "lr": .006, "l2": 0.0001, "dropout": 0.20, "batch": 128}),
        ("manual_4_two_hidden", {"hidden": [576, 288], "lr": .008, "l2": 0.00005, "dropout": 0.10, "batch": 128}),
    ]


def random_candidates(seed_result, n_trials, quick=False):
    """수동 탐색에서 성능 좋았던 구조 근처를 무작위로 탐색."""
    rng = np.random.default_rng(SEED + 2026)
    reference = seed_result["config"]["hidden"]
    results = []
    for i in range(n_trials):
        if quick:
            cfg = {"hidden": [32, 16], "lr": .008, "l2": 0.0, "dropout": 0.0, "batch": 128}
        else:
            scale = float(rng.choice([0.8, 1.0, 1.2]))
            widths = [min(640, max(64, int(round(w * scale / 16) * 16))) for w in reference]
            cfg = {
                "hidden": widths,
                "lr": float(10 ** rng.uniform(np.log10(.003), np.log10(.018))),
                "l2": float(rng.choice([0.0, 1e-5, 5e-5, 1e-4, 3e-4])),
                "dropout": float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])),
                "batch": int(rng.choice([64, 128, 256])),
            }
        results.append((f"random_{i + 1:02d}", cfg))
    return results


def ranking(row):
    # 후보 간 선택은 검증 정확도 우선, 동점이면 검증 교차 엔트로피.
    return (-row["val_accuracy"], row["val_ce"])


def plot_selected(history_path, output_dir):
    with history_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]
    for metric, fname, ylabel in (
        (("train_sample_accuracy", "val_accuracy"), "accuracy.png", "Accuracy"),
        (("train_sample_ce", "val_ce"), "loss.png", "Cross entropy"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for col in metric:
            ax.plot(epochs, [float(r[col]) for r in rows], label=col)
        ax.set(xlabel="Epoch", ylabel=ylabel, title="Selected AdaGrad model")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=160)
        plt.close(fig)


def run_search(run_dir, quick=False, trials=5, manual_epochs=12,
               random_epochs=16, final_epochs=32):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "plan.json"
    config = {"quick": quick, "trials": trials, "manual_epochs": manual_epochs,
              "random_epochs": random_epochs, "final_epochs": final_epochs, "seed": SEED}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior != config:
            raise ValueError("기존 run의 설정과 다릅니다. 새 run 폴더로 실행하세요.")
    else:
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"결과 폴더: {run_dir}", flush=True)
    print(f'중단 후: python fashion/train_adagrad_best.py --resume "{run_dir}"', flush=True)
    x, labels = load_data("train")  # 이 함수는 테스트 파일을 읽지 않음
    dataset = train_validation_split(x, labels, quick)
    print(f"학습 {len(dataset[1]):,}장 / 검증 {len(dataset[3]):,}장", flush=True)
    results = []
    for name, cfg in manual_configs(quick):
        result = train_trial(name, cfg, 2 if quick else manual_epochs,
                             dataset, run_dir / "manual" / name, quick)
        results.append(result)
        write_csv(run_dir / "manual_results.csv", results)

    manual_best = min(results, key=ranking)
    random_plan_path = run_dir / "random_plan.json"
    if random_plan_path.exists():
        randoms = json.loads(random_plan_path.read_text(encoding="utf-8"))
    else:
        randoms = random_candidates(manual_best, 1 if quick else trials, quick)
        random_plan_path.write_text(json.dumps(randoms, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, cfg in randoms:
        result = train_trial(name, cfg, 2 if quick else random_epochs,
                             dataset, run_dir / "random" / name, quick)
        results.append(result)
        write_csv(run_dir / "all_screening_results.csv", sorted(results, key=ranking))

    # 최종 후보 2개를 동일한 32에폭 예산으로 재평가. 빠른 작동 검사 시에는 1개.
    unique = []
    seen = set()
    for result in sorted(results, key=ranking):
        key = json.dumps(result["config"], sort_keys=True)
        if key not in seen:
            unique.append(result)
            seen.add(key)
    finalists = unique[:(1 if quick else 2)]
    final_results = []
    for i, previous in enumerate(finalists, start=1):
        name = f"final_{i}"
        result = train_trial(name, previous["config"], 2 if quick else final_epochs,
                             dataset, run_dir / "final" / name, quick)
        final_results.append(result)
        write_csv(run_dir / "final_results.csv", sorted(final_results, key=ranking))

    # 장기 학습이 더 나쁘면 스크리닝 최적 체크포인트를 유지한다.
    selected = min(results + final_results, key=ranking)
    if selected["name"].startswith("manual"):
        source = run_dir / "manual" / selected["name"] / "best_model.pkl"
        history = run_dir / "manual" / selected["name"] / "history.csv"
    elif selected["name"].startswith("random"):
        source = run_dir / "random" / selected["name"] / "best_model.pkl"
        history = run_dir / "random" / selected["name"] / "history.csv"
    else:
        source = run_dir / "final" / selected["name"] / "best_model.pkl"
        history = run_dir / "final" / selected["name"] / "history.csv"

    target_dir = run_dir / "Competition"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / ("QUICK_ONLY_DO_NOT_SUBMIT.pkl" if quick else "network_Team6.pkl")
    target.write_bytes(source.read_bytes())
    plot_selected(history, run_dir)
    output = {"selected_name": selected["name"], "selected_config": selected["config"],
              "selected_epoch": selected["best_epoch"],
              "selected_val_accuracy": selected["val_accuracy"],
              "selected_val_ce": selected["val_ce"], "test_evaluated": False,
              "quick": quick, "model_path": str(target.resolve()),
              "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (run_dir / "selection.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== 검증 기준으로 선택된 단일 AdaGrad 모델 ===", flush=True)
    print(f"설정: {selected['config']} | epoch={selected['best_epoch']}", flush=True)
    print(f"검증 정확도: {selected['val_accuracy']:.2%}, 검증 CE: {selected['val_ce']:.4f}", flush=True)
    print(f"저장 위치: {target}", flush=True)
    if quick:
        print("빠른 점검 결과는 제출 성능으로 사용하면 안 됩니다.", flush=True)
    else:
        print("테스트 데이터는 아직 사용하지 않았습니다. 최종 설정 확정 후에만 --evaluate를 실행하세요.")


def evaluate_final(run_dir):
    selected_path = run_dir / "selection.json"
    if not selected_path.exists():
        raise FileNotFoundError("먼저 --full로 학습 및 모델 선택을 완료하세요.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["quick"]:
        raise ValueError("--quick 결과는 테스트 평가/제출에 사용할 수 없습니다.")
    existing = run_dir / "final_test.json"
    if existing.exists():
        print("이미 기록된 최종 테스트 결과입니다. 다시 측정하지 않습니다.")
        print(existing.read_text(encoding="utf-8"))
        return
    model_path = Path(selected["model_path"])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != selected["model_sha256"]:
        raise ValueError("선택된 모델 파일이 변경되었습니다.")
    # pickle은 자신이 생성했거나 신뢰할 수 있는 파일만 열어야 합니다.
    with model_path.open("rb") as f:
        net = pickle.load(f)
    x_test, t_test = load_data("t10k")  # 유일한 최종 시험 데이터 사용 지점
    test_acc, test_ce = accuracy_loss(net, x_test, t_test)
    outcome = {"test_accuracy": test_acc, "test_ce": test_ce,
               "count": len(t_test), "model_sha256": selected["model_sha256"]}
    existing.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(f"최종 테스트 정확도: {test_acc:.2%}, 테스트 CE: {test_ce:.4f}")
    print("테스트 결과를 보고 다시 모델을 고르면 독립적인 최종 평가가 깨집니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="2,000장/2에폭 기능 점검")
    mode.add_argument("--full", action="store_true", help="수동→랜덤→장기학습")
    mode.add_argument("--resume", type=Path, help="이전 run 폴더 재개")
    mode.add_argument("--evaluate", type=Path, help="선택이 끝난 run의 최종 테스트 1회")
    parser.add_argument("--trials", type=int, default=5, help="Randomized Search 횟수")
    parser.add_argument("--manual-epochs", type=int, default=12)
    parser.add_argument("--random-epochs", type=int, default=16)
    parser.add_argument("--final-epochs", type=int, default=32)
    args = parser.parse_args()
    if args.evaluate:
        evaluate_final(args.evaluate.resolve())
        return
    if args.resume:
        run_dir = args.resume.resolve()
        if (run_dir / "selection.json").exists():
            print("이 실행은 이미 완료되었습니다:", run_dir)
            print("새 결과를 원하면 --full로 새 실행을 시작하세요.")
            return
        plan_path = run_dir / "plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"run 폴더에 plan.json이 없습니다: {run_dir}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        run_search(run_dir, quick=plan["quick"], trials=plan["trials"],
                   manual_epochs=plan["manual_epochs"], random_epochs=plan["random_epochs"],
                   final_epochs=plan["final_epochs"])
        return
    if min(args.trials, args.manual_epochs, args.random_epochs, args.final_epochs) < 1:
        parser.error("탐색 횟수·에폭 수는 모두 1 이상이어야 합니다.")
    prefix = "quick" if args.quick else "full"
    run_dir = HERE / "adagrad_runs" / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_search(run_dir, quick=args.quick, trials=args.trials,
               manual_epochs=args.manual_epochs, random_epochs=args.random_epochs,
               final_epochs=args.final_epochs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n학습 중단. 완료된 실험은 저장되었으며, 출력된 --resume 명령으로 재개하세요.")
        sys.exit(130)
