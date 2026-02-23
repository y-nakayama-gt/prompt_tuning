import argparse
import os
from pathlib import Path
from typing import Any

import dspy
import yaml
from dotenv import load_dotenv

from src.evaluation.metrics import make_image_reproduction_metric
from src.modules.pipelines import ImageDescriptionPipeline

load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """設定ファイルを読み込む。

    Args:
        config_path: 設定ファイルのパス。

    Returns:
        設定辞書。
    """
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def load_examples(images_dir: str = "data/images") -> list[dspy.Example]:
    """画像ディレクトリからDSPy Exampleのリストを構築する。

    Args:
        images_dir: 画像が格納されているディレクトリのパス。

    Returns:
        DSPy Example のリスト。各Exampleはimage と image_path フィールドを持つ。
    """
    images_path = Path(images_dir)
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    examples = []
    for p in sorted(images_path.iterdir()):
        if p.suffix.lower() in supported_extensions:
            example = dspy.Example(
                image=dspy.Image(str(p)),
                image_path=str(p),
            ).with_inputs("image")
            examples.append(example)
    return examples


def split_dataset(
    examples: list[dspy.Example],
    train_ratio: float = 2 / 3,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """データセットをtrainとvalに分割する。

    Args:
        examples: 全Exampleのリスト。
        train_ratio: トレーニングセットの割合。

    Returns:
        (trainset, valset) のタプル。
    """
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


def build_optimizer(
    optimizer_name: str,
    metric: Any,
    config: dict,
) -> Any:
    """オプティマイザを名前から生成する。

    Args:
        optimizer_name: オプティマイザ名。"mipro", "bootstrap", "copro" のいずれか。
        metric: 評価メトリクス関数。
        config: 設定辞書。

    Returns:
        DSPy オプティマイザインスタンス。

    Raises:
        ValueError: 未知のオプティマイザ名が指定された場合。
    """
    opt_cfg = config["optimizer"]
    if optimizer_name == "mipro":
        return dspy.MIPROv2(
            metric=metric,
            num_candidates=opt_cfg["num_candidates"],
            num_threads=opt_cfg["num_threads"],
            auto=None,
        )
    if optimizer_name == "bootstrap":
        return dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=opt_cfg["max_bootstrapped_demos"],
            max_labeled_demos=opt_cfg["max_labeled_demos"],
            max_rounds=1,
        )
    if optimizer_name == "copro":
        return dspy.COPRO(
            metric=metric,
            depth=opt_cfg.get("copro_depth", 3),
            breadth=opt_cfg.get("copro_breadth", 5),
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from: mipro, bootstrap, copro")


def compile_pipeline(
    optimizer_name: str,
    optimizer: Any,
    pipeline: ImageDescriptionPipeline,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    config: dict,
) -> Any:
    """オプティマイザに応じて compile を呼び分ける。

    Args:
        optimizer_name: オプティマイザ名。
        optimizer: オプティマイザインスタンス。
        pipeline: 最適化対象のパイプライン。
        trainset: 訓練データ。
        valset: 検証データ。
        config: 設定辞書。

    Returns:
        最適化済みパイプライン。
    """
    opt_cfg = config["optimizer"]
    if optimizer_name == "mipro":
        minibatch_size = min(opt_cfg["minibatch_size"], len(valset))
        return optimizer.compile(
            pipeline,
            trainset=trainset,
            valset=valset,
            num_trials=opt_cfg["num_trials"],
            max_bootstrapped_demos=opt_cfg["max_bootstrapped_demos"],
            max_labeled_demos=opt_cfg["max_labeled_demos"],
            minibatch_size=minibatch_size,
            minibatch_full_eval_steps=opt_cfg["minibatch_full_eval_steps"],
        )
    if optimizer_name == "bootstrap":
        return optimizer.compile(pipeline, trainset=trainset)
    if optimizer_name == "copro":
        return optimizer.compile(pipeline, trainset=trainset, eval_kwargs={"num_threads": opt_cfg["num_threads"]})
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main() -> None:
    """画像説明プロンプト最適化のメインエントリポイント。"""
    parser = argparse.ArgumentParser(description="画像説明プロンプト最適化")
    parser.add_argument(
        "-op",
        "--optimizer",
        choices=["mipro", "bootstrap", "copro"],
        default="mipro",
        help="使用するオプティマイザ (default: mipro)",
    )
    args = parser.parse_args()

    config = load_config()

    # DSPy LM設定
    lm = dspy.LM(
        model=config["vision_lm"]["model"],
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=config["vision_lm"]["max_tokens"],
    )
    dspy.configure(lm=lm)

    # データセット読み込み
    examples = load_examples()
    if len(examples) == 0:
        print("data/images/ に画像ファイルが見つかりません。画像を配置してから再実行してください。")
        return

    trainset, valset = split_dataset(examples)
    print(f"オプティマイザ: {args.optimizer}")
    print(f"データセット: train={len(trainset)}, val={len(valset)}")

    # メトリクス構築
    metric = make_image_reproduction_metric(
        gen_model=config["image_generation"]["model"],
        gen_size=config["image_generation"]["size"],
        eval_model=config["evaluation"]["model"],
    )

    # 最適化
    pipeline = ImageDescriptionPipeline()
    optimizer = build_optimizer(args.optimizer, metric, config)
    optimized_pipeline = compile_pipeline(args.optimizer, optimizer, pipeline, trainset, valset, config)

    # 保存
    experiments_dir = Path("experiments")
    experiments_dir.mkdir(exist_ok=True)
    save_path = experiments_dir / f"optimized_pipeline_{args.optimizer}.json"
    optimized_pipeline.save(str(save_path))
    print(f"最適化済みパイプラインを保存しました: {save_path}")


if __name__ == "__main__":
    main()
