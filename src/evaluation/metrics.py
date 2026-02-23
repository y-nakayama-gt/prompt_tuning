from collections.abc import Callable
from typing import Any

from openai import OpenAI

from src.evaluation.image_generator import generate_image, image_file_to_base64


def compare_images(
    original_b64: str,
    original_mime: str,
    generated_b64: str,
    eval_model: str,
) -> float:
    """2枚の画像をvision LLMで比較し、類似度スコアを返す。

    Args:
        original_b64: 元画像のbase64エンコード文字列。
        original_mime: 元画像のMIMEタイプ。
        generated_b64: 生成画像のbase64エンコード文字列。
        eval_model: 評価に使用するvisionモデル名。

    Returns:
        0.0〜1.0の類似度スコア。
    """
    client = OpenAI()
    prompt = (
        "以下の2枚の画像を比較し、どれほど似ているかを0.0〜1.0のスコアで評価してください。"
        "1枚目が元画像、2枚目が再生成された画像です。"
        "被写体、構図、色調、スタイルなどを総合的に評価し、"
        "スコアのみを数値(例: 0.75)で返してください。他のテキストは一切含めないでください。"
    )
    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{original_mime};base64,{original_b64}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{generated_b64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip()
    try:
        score = float(raw)
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.0


def make_image_reproduction_metric(
    gen_model: str,
    gen_size: str,
    eval_model: str,
) -> Callable[[Any, Any, Any], float]:
    """画像再現性メトリクス関数を生成するファクトリ。

    生成された説明文を使って画像を再生成し、元画像との類似度を評価する。

    Args:
        gen_model: 画像生成に使用するモデル名。
        gen_size: 生成画像のサイズ。
        eval_model: 評価に使用するvisionモデル名。

    Returns:
        DSPy metric関数 (example, prediction, trace) -> float。
    """

    def metric(example: Any, prediction: Any, trace: Any = None) -> float:
        """DSPy用メトリクス関数。

        Args:
            example: image_path フィールドを持つ DSPy Example。
            prediction: description フィールドを持つ DSPy Prediction。
            trace: DSPy トレース（未使用）。

        Returns:
            0.0〜1.0の類似度スコア。
        """
        original_b64, original_mime = image_file_to_base64(example.image_path)
        generated_b64 = generate_image(
            prompt=prediction.description,
            model=gen_model,
            size=gen_size,
        )
        return compare_images(
            original_b64=original_b64,
            original_mime=original_mime,
            generated_b64=generated_b64,
            eval_model=eval_model,
        )

    return metric
