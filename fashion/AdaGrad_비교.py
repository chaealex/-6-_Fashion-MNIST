"""처음 코드 설정(은닉층 100×3, AdaGrad)에 L2와 드롭아웃을 적용한 실험."""

from NeuralNet_fashion import preset_cli


if __name__ == "__main__":
    preset_cli(
        "수동",
        1,
        config_overrides={
            "hidden": [100, 100, 100],
            "optimizer": "adagrad",
            "lr": 0.01,
            "l2": 2e-5,        # L2 가중치 감소 (공용 코드 기본값)
            "dropout": 0.15,   # 드롭아웃 비율 (공용 코드 기본값)
        },
        title="AdaGrad + L2·드롭아웃",
    )