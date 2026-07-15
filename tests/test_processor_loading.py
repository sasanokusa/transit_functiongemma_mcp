import unittest
from unittest.mock import Mock

from transit_functiongemma.processor import load_router_processor


class ProcessorLoadingTest(unittest.TestCase):
    def test_adapter_processor_is_used_when_available(self) -> None:
        load = Mock()
        processor = object()
        load.return_value = processor
        self.assertIs(
            load_router_processor(
                "base-model", "adapter", processor_loader=load
            ),
            processor,
        )
        load.assert_called_once_with("adapter")

    def test_missing_adapter_processor_falls_back_to_base(self) -> None:
        load = Mock()
        processor = object()
        load.side_effect = [OSError("missing config.json"), processor]
        self.assertIs(
            load_router_processor(
                "base-model", "adapter", processor_loader=load
            ),
            processor,
        )
        self.assertEqual(
            [call.args for call in load.call_args_list],
            [("adapter",), ("base-model",)],
        )

    def test_base_processor_failure_is_not_hidden(self) -> None:
        load = Mock()
        load.side_effect = OSError("base unavailable")
        with self.assertRaisesRegex(OSError, "base unavailable"):
            load_router_processor("base-model", processor_loader=load)
        load.assert_called_once_with("base-model")


if __name__ == "__main__":
    unittest.main()
