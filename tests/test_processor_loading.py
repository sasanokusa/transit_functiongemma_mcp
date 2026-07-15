import unittest
from unittest.mock import patch

from transit_functiongemma.infer import load_router_processor


class ProcessorLoadingTest(unittest.TestCase):
    @patch("transit_functiongemma.infer.AutoProcessor.from_pretrained")
    def test_adapter_processor_is_used_when_available(self, load) -> None:
        processor = object()
        load.return_value = processor
        self.assertIs(load_router_processor("base-model", "adapter"), processor)
        load.assert_called_once_with("adapter")

    @patch("transit_functiongemma.infer.AutoProcessor.from_pretrained")
    def test_missing_adapter_processor_falls_back_to_base(self, load) -> None:
        processor = object()
        load.side_effect = [OSError("missing config.json"), processor]
        self.assertIs(load_router_processor("base-model", "adapter"), processor)
        self.assertEqual(
            [call.args for call in load.call_args_list],
            [("adapter",), ("base-model",)],
        )

    @patch("transit_functiongemma.infer.AutoProcessor.from_pretrained")
    def test_base_processor_failure_is_not_hidden(self, load) -> None:
        load.side_effect = OSError("base unavailable")
        with self.assertRaisesRegex(OSError, "base unavailable"):
            load_router_processor("base-model")
        load.assert_called_once_with("base-model")


if __name__ == "__main__":
    unittest.main()
