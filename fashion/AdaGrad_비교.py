"""실험 정리의 작은 모델 A: L2와 드롭아웃 미적용."""

from NeuralNet_fashion import preset_cli


if __name__ == "__main__":
    preset_cli(
        "수동",
        1,
        config_overrides={
            "hidden": [100, 100, 100],
            "optimizer": "adagrad",
            "lr": 0.01,
            "l2": 0.0,
            "dropout": 0.0,
        },
        title="AdaGrad 작은 모델 (L2·드롭아웃 미적용)",
    )
