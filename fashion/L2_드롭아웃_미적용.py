"""자동 1과 같은 설정에서 L2와 드롭아웃을 제외한 비교 실험."""

from NeuralNet_fashion import preset_cli


if __name__ == "__main__":
    preset_cli(
        "랜덤",
        1,
        config_overrides={"l2": 0.0, "dropout": 0.0},
        title="L2·드롭아웃 미적용",
    )
