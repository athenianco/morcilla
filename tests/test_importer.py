import pytest

from morcilla.importer import ImportFromStringError, import_from_string


def test_invalid_format():
    with pytest.raises(ImportFromStringError) as exc_info:
        import_from_string("example:")
    expected = 'Import string "example:" must be in format "<module>:<attribute>".'
    assert exc_info.match(expected)


def test_invalid_module():
    with pytest.raises(ImportFromStringError) as exc_info:
        import_from_string("module_does_not_exist:myattr")
    expected = 'Could not import module "module_does_not_exist".'
    assert exc_info.match(expected)


def test_invalid_attr():
    with pytest.raises(ImportFromStringError) as exc_info:
        import_from_string("tempfile:attr_does_not_exist")
    expected = 'Attribute "attr_does_not_exist" not found in module "tempfile".'
    assert exc_info.match(expected)


def test_internal_import_error():
    with pytest.raises(ImportError):
        import_from_string("tests.importer.raise_import_error:myattr")


def test_valid_import():
    instance = import_from_string("tempfile:TemporaryFile")
    from tempfile import TemporaryFile

    assert instance == TemporaryFile
