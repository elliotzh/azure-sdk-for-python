import re

import pytest


class MockClass:
    mock_cls_attr = 2

    def __init__(self, mock_arg):
        self.mock_arg = mock_arg

    def mock_instance_func(self, arg):
        result = self.mock_arg + arg
        return result

    @classmethod
    def mock_class_method(cls, arg):
        result = cls.mock_cls_attr + arg
        return result

    def __call__(self, arg):
        result = self.mock_arg + arg
        return result


def mock_conflict_function(__self):
    return __self


def mock_function_with_self(self):
    return self


@pytest.mark.unittest
@pytest.mark.pipeline_test
class TestPersistentLocals:
    def get_output_and_locals(self, func, injected_params):
        from azure.ai.ml._utils._func_utils import _get_outputs_and_locals
        return _get_outputs_and_locals(func, injected_params)

    def test_simple(self):
        def mock_function(mock_arg):
            mock_local_variable = 1
            return mock_local_variable, mock_arg

        outputs, _locals = self.get_output_and_locals(mock_function, {"mock_arg": 1})
        assert outputs == (1, 1)
        assert set(_locals.keys()) == {'mock_arg', 'mock_local_variable'}

    def test_func_with_named_self_argument(self):
        outputs, _locals = self.get_output_and_locals(mock_function_with_self, {"self": 1})
        assert outputs == 1
        assert set(_locals.keys()) == {'self'}

    def test_raise_exception(self):
        def mock_error_exception():
            mock_local_variable = 1
            return mock_local_variable / 0

        with pytest.raises(ZeroDivisionError):
            self.get_output_and_locals(mock_error_exception, {})

    def test_instance_func(self):
        mock_obj = MockClass(1)
        outputs, _locals = self.get_output_and_locals(mock_obj.mock_instance_func, {"arg": 1})
        assert outputs == 2
        assert set(_locals.keys()) == {'result', 'arg', 'self'}

    def test_class_method(self):
        mock_obj = MockClass(1)
        outputs, _locals = self.get_output_and_locals(mock_obj.mock_class_method, {"arg": 1})
        assert outputs == 3
        assert set(_locals.keys()) == {'result', 'arg', 'cls'}

    def test_instance_call(self):
        mock_obj = MockClass(1)
        outputs, _locals = self.get_output_and_locals(mock_obj, {"arg": 1})
        assert outputs == 2
        assert set(_locals.keys()) == {'result', 'arg', 'self'}

    def test_invalid_passed_func(self):
        with pytest.raises(TypeError, match='func must be a function or a callable object'):
            self.get_output_and_locals(1, {"arg": 1})

    def test_param_conflict(self):
        with pytest.raises(
            ValueError,
            match=re.escape('Injected param name __self conflicts with function args [\'__self\']'),
        ):
            self.get_output_and_locals(mock_conflict_function, {"arg": 1})


@pytest.mark.unittest
@pytest.mark.pipeline_test
@pytest.mark.usefixtures('enable_pipeline_private_preview_features')
class TestPersistentLocalsPrivatePreview(TestPersistentLocals):
    def get_output_and_locals(self, func, injected_params):
        from azure.ai.ml._utils._func_utils import _get_outputs_and_locals_private_preview
        return _get_outputs_and_locals_private_preview(func, injected_params)

    @pytest.fixture(scope='class')
    def install_bytecode(self):
        import pip
        pip.main(['install', 'bytecode'])
