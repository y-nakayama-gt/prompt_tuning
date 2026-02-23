import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


def generate_image(prompt: str, model: str, size: str) -> str:
    """テキストプロンプトから画像を生成し、base64文字列を返す。

    Args:
        prompt: 画像生成プロンプト。
        model: 使用する画像生成モデル名。
        size: 生成画像のサイズ（例: "1024x1024"）。

    Returns:
        生成画像のbase64エンコード文字列。
    """
    client = OpenAI()
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=1,
    )
    return response.data[0].b64_json


def image_file_to_base64(image_path: str) -> tuple[str, str]:
    """画像ファイルをbase64文字列とMIMEタイプに変換する。

    Args:
        image_path: 画像ファイルのパス。

    Returns:
        (base64エンコード文字列, MIMEタイプ) のタプル。
    """
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/jpeg"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime_type
