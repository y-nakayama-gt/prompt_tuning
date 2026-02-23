from unittest.mock import MagicMock

import dspy
import pytest

from src.modules.pipelines import ImageDescriptionPipeline


@pytest.fixture
def pipeline() -> ImageDescriptionPipeline:
    """テスト用パイプラインインスタンスを生成する。"""
    return ImageDescriptionPipeline()


def test_pipeline_forward_returns_description(pipeline: ImageDescriptionPipeline) -> None:
    """forwardメソッドがdescriptionフィールドを持つPredictionを返すことを確認する。"""
    mock_prediction = MagicMock()
    mock_prediction.description = "A beautiful sunset over the ocean"

    pipeline.describe = MagicMock(return_value=mock_prediction)
    dummy_image = MagicMock(spec=dspy.Image)
    result = pipeline(image=dummy_image)

    assert result.description == "A beautiful sunset over the ocean"


def test_pipeline_calls_describe_with_image(pipeline: ImageDescriptionPipeline) -> None:
    """forwardメソッドがimageを引数にdescribeを呼び出すことを確認する。"""
    mock_prediction = MagicMock()
    mock_prediction.description = "A cat sitting on a chair"
    dummy_image = MagicMock(spec=dspy.Image)

    pipeline.describe = MagicMock(return_value=mock_prediction)
    pipeline(image=dummy_image)

    pipeline.describe.assert_called_once_with(image=dummy_image)


def test_pipeline_has_describe_module(pipeline: ImageDescriptionPipeline) -> None:
    """パイプラインがdescribeモジュールを持つことを確認する。"""
    assert hasattr(pipeline, "describe")
    assert isinstance(pipeline.describe, dspy.ChainOfThought)
