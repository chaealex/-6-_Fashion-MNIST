from NeuralNet_fashion import preset_cli


if __name__ == "__main__":
    preset_cli(
        "수동",
        1,
        config_overrides={
            "hidden": [256, 128, 64],
            "optimizer": "adagrad",
            "lr": 0.05,
            "l2": 2e-5,
            "dropout": 0.15,
        },
        title="AdaGrad 큰 모델 (L2·드롭아웃)",
    )