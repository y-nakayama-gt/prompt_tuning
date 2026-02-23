from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.image_generator import generate_image, image_file_to_base64
from src.evaluation.metrics import compare_images, make_image_reproduction_metric


class TestGenerateImage:
    """generate_image関数のテスト。"""

    def test_returns_base64_string(self) -> None:
        """OpenAI APIを呼び出してbase64文字列を返すことを確認する。"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="dGVzdA==")]

        with patch("src.evaluation.image_generator.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.images.generate.return_value = mock_response

            result = generate_image(prompt="a cat", model="gpt-image-1-mini", size="1024x1024")

        assert result == "dGVzdA=="

    def test_calls_api_with_correct_params(self) -> None:
        """正しいパラメータでOpenAI APIを呼び出すことを確認する。"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="abc123")]

        with patch("src.evaluation.image_generator.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.images.generate.return_value = mock_response

            generate_image(prompt="a dog", model="test-model", size="512x512")

            mock_client.images.generate.assert_called_once_with(
                model="test-model",
                prompt="a dog",
                size="512x512",
                response_format="b64_json",
                n=1,
            )


class TestImageFileToBase64:
    """image_file_to_base64関数のテスト。"""

    def test_returns_tuple_of_b64_and_mime(self, tmp_path: pytest.TempPathFactory) -> None:
        """(base64文字列, MIMEタイプ) のタプルを返すことを確認する。"""
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        b64, mime = image_file_to_base64(str(image_file))

        assert isinstance(b64, str)
        assert mime == "image/png"

    def test_jpeg_mime_type(self, tmp_path: pytest.TempPathFactory) -> None:
        """JPEGファイルのMIMEタイプが正しく検出されることを確認する。"""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"\xff\xd8\xff")

        _, mime = image_file_to_base64(str(image_file))

        assert mime == "image/jpeg"

    def test_unknown_extension_defaults_to_jpeg(self, tmp_path: pytest.TempPathFactory) -> None:
        """不明な拡張子の場合、MIMEタイプがimage/jpegにフォールバックすることを確認する。"""
        image_file = tmp_path / "test.unknownext"
        image_file.write_bytes(b"dummy")

        _, mime = image_file_to_base64(str(image_file))

        assert mime == "image/jpeg"


class TestCompareImages:
    """compare_images関数のテスト。"""

    def test_returns_float_score(self) -> None:
        """float型のスコアを返すことを確認する。"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"

        with patch("src.evaluation.metrics.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = mock_response

            score = compare_images("b64orig", "image/png", "b64gen", "gpt-4o-mini")

        assert score == pytest.approx(0.85)

    def test_score_clamped_to_zero_on_invalid_response(self) -> None:
        """不正なレスポンスの場合、スコアが0.0にフォールバックすることを確認する。"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not a number"

        with patch("src.evaluation.metrics.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = mock_response

            score = compare_images("b64orig", "image/png", "b64gen", "gpt-4o-mini")

        assert score == pytest.approx(0.0)

    def test_score_clamped_above_one(self) -> None:
        """スコアが1.0を超える場合、1.0にクランプされることを確認する。"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "1.5"

        with patch("src.evaluation.metrics.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = mock_response

            score = compare_images("b64orig", "image/png", "b64gen", "gpt-4o-mini")

        assert score == pytest.approx(1.0)


class TestMakeImageReproductionMetric:
    """make_image_reproduction_metric関数のテスト。"""

    def test_returns_callable(self) -> None:
        """Callable を返すことを確認する。"""
        metric = make_image_reproduction_metric("gpt-image-1-mini", "1024x1024", "gpt-4o-mini")
        assert callable(metric)

    def test_metric_calls_components(self) -> None:
        """metricがimage_file_to_base64, generate_image, compare_imagesを呼び出すことを確認する。"""
        metric: Callable = make_image_reproduction_metric(
            "gpt-image-1-mini", "1024x1024", "gpt-4o-mini"
        )

        example = MagicMock()
        example.image_path = "/path/to/image.png"

        prediction = MagicMock()
        prediction.description = "A beautiful landscape"

        with (
            patch(
                "src.evaluation.metrics.image_file_to_base64",
                return_value=("orig_b64", "image/png"),
            ) as mock_file_to_b64,
            patch(
                "src.evaluation.metrics.generate_image",
                return_value="gen_b64",
            ) as mock_gen,
            patch(
                "src.evaluation.metrics.compare_images",
                return_value=0.9,
            ) as mock_compare,
        ):
            score = metric(example, prediction)

        mock_file_to_b64.assert_called_once_with("/path/to/image.png")
        mock_gen.assert_called_once_with(
            prompt="A beautiful landscape",
            model="gpt-image-1-mini",
            size="1024x1024",
        )
        mock_compare.assert_called_once_with(
            original_b64="orig_b64",
            original_mime="image/png",
            generated_b64="gen_b64",
            eval_model="gpt-4o-mini",
        )
        assert score == pytest.approx(0.9)
