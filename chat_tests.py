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

    def test_docs_folder_is_preferred_over_docs_not(self):
        module = load_chat_module()
        with patch(
            "os.path.isdir",
            side_effect=lambda path: path == "docs" or path == "docs_not",
        ), patch("os.listdir", return_value=["vocabulary.txt"]):
            self.assertEqual(module.resolve_docs_dir(), "docs")

    def test_main_exits_immediately(self):
        module = load_chat_module()
        with patch("builtins.input", side_effect=["exit"]):
            with patch("builtins.print"):
                module.main()

    def test_extract_keywords_filters_stop_words_and_duplicates(self):
        module = load_chat_module()
        self.assertEqual(
            module.extract_keywords(
                "Projekt aplikacji dla uczniów, aplikacji mobilnej i szkół"
            ),
            ["Projekt", "aplikacji", "uczniów"],
        )
        self.assertEqual(
            module.extract_keywords("Build a useful learning platform"),
            ["Build", "useful", "learning"],
        )
        self.assertEqual(module.extract_keywords("No keywords here", limit=0), [])

    def test_main_prints_keywords_for_external_and_internal_ideas(self):
        module = load_chat_module()
        module.docs = ["Projekt aplikacji mobilnej dla uczniów"]
        module.listen_docs = ["learning", "vocabulary"]
        module.banned_words = []
        module.qa_generator = None
        module.idea_generator = None

        printed = []
        with patch("builtins.input", side_effect=["projekt aplikacji", "exit"]):
            with patch.object(
                module,
                "_find_best_context",
                return_value=("Projekt aplikacji mobilnej dla uczniów", []),
            ):
                with patch("builtins.print", side_effect=lambda *args: printed.append(args)):
                    module.main()

        output = "\n".join(" ".join(map(str, line)) for line in printed)
        self.assertIn("Pomysły_zewnętrzne (PL)", output)
        self.assertIn("Pomysły_wewnętrzne (PL)", output)
        self.assertIn("biblioteka", output)
        self.assertIn("współpraca z docs_listen i tekstem użytkownika", output)
        self.assertGreaterEqual(output.count("Słowa kluczowe:"), 4)

    def test_idea_context_combines_listen_words_question_and_speech_answer(self):
        module = load_chat_module()
        module.listen_docs = ["travel vocabulary", "airport terms"]

        context = module._build_idea_context("Plan a trip", "Use the airport terminal")

        self.assertIn("travel vocabulary", context)
        self.assertIn("airport terms", context)
        self.assertIn("Plan a trip", context)
        self.assertIn("Use the airport terminal", context)

    def test_idea_group_uses_one_english_mode(self):
        module = load_chat_module()
        calls = []

        def fake_generate(context, language="pl", limit=3, mode=0):
            calls.append((context, language, mode))
            return ["Idea"]

        module.generate_ideas = fake_generate
        with patch("builtins.print"):
            module._print_idea_group("biblioteka", "all docs", "one doc")

        english_modes = [mode for _, language, mode in calls if language == "en"]
        self.assertEqual(english_modes, [0, 0])
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
