import dspy

from src.modules.signatures import ImageDescriptionSignature


class ImageDescriptionPipeline(dspy.Module):
    """画像を詳細なテキスト説明に変換するパイプライン"""

    def __init__(self) -> None:
        super().__init__()
        self.describe = dspy.ChainOfThought(ImageDescriptionSignature)

    def forward(self, image: dspy.Image) -> dspy.Prediction:
        """画像を受け取り、説明文を生成する。

        Args:
            image: 説明対象の画像。

        Returns:
            description フィールドを持つ Prediction オブジェクト。
        """
        return self.describe(image=image)
