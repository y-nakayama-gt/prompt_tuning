import dspy


class ImageDescriptionSignature(dspy.Signature):
    """元画像と同じ画像が再生成できるよう、画像を詳細に説明する"""

    image: dspy.Image = dspy.InputField(description="説明する画像")
    description: str = dspy.OutputField(description="画像の詳細な説明文")
