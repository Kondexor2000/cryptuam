import importlib.util
import sys
import types
import unittest
from unittest.mock import mock_open, patch

import numpy as np

MODULE_PATH = r"c:\Users\kondz\OneDrive\Pulpit\Moje Prace\ChatGPT\chat.py"


def _make_transformers_module():
    module = types.ModuleType("transformers")

    class FakeTokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = None

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

    class FakeModel:
        pass

    def pipeline(task, model, tokenizer):
        class FakePipeline:
            def __call__(self, prompt, *args, **kwargs):
                if "Pomysły" in prompt:
                    return [{"generated_text": prompt + "Idea 1\nIdea 2"} for _ in range(3)]
                return [{"generated_text": "Test answer"}]

        return FakePipeline()

    module.AutoTokenizer = FakeTokenizer
    module.AutoModelForSeq2SeqLM = types.SimpleNamespace(from_pretrained=lambda name: FakeModel())
    module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda name: FakeModel())
    module.pipeline = pipeline
    return module


def _make_faiss_module():
    module = types.ModuleType("faiss")

    class FakeIndex:
        def search(self, q, k):
            return None, np.array([[0, 0, 0, 0, 0]])

    module.read_index = lambda path: FakeIndex()
    module.normalize_L2 = lambda x: x
    return module


def _make_hub_module():
    module = types.ModuleType("tensorflow_hub")

    class FakeEmbedder:
        def __call__(self, texts):
            class Out:
                def numpy(self):
                    return np.array([[1.0, 0.0, 0.0]])

            return Out()

    module.load = lambda url: FakeEmbedder()
    return module


def load_chat_module():
    sys.modules["faiss"] = _make_faiss_module()
    sys.modules["tensorflow"] = types.ModuleType("tensorflow")
    sys.modules["tensorflow_hub"] = _make_hub_module()
    sys.modules["transformers"] = _make_transformers_module()

    with patch("numpy.load", return_value=np.array(["doc1", "doc2", "doc3"])):
        with patch("os.listdir", return_value=["banned.txt"]):
            with patch("builtins.open", mock_open(read_data="bad\n")):
                spec = importlib.util.spec_from_file_location("chat", MODULE_PATH)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
    return module


class ChatTests(unittest.TestCase):
    def test_banned_words_are_loaded(self):
        module = load_chat_module()
        self.assertEqual(module.banned_words, ["bad"])

    def test_main_exits_immediately(self):
        module = load_chat_module()
        with patch("builtins.input", side_effect=["exit"]):
            with patch("builtins.print"):
                module.main()


if __name__ == "__main__":
    unittest.main()
