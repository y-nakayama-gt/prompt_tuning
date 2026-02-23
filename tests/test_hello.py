from prompt_tuning import hello


def test_hello() -> None:
    result = hello()
    expected = "Hello from prompt_tuning!"
    assert result == expected


def test_hello_return_type() -> None:
    result = hello()
    assert isinstance(result, str)
