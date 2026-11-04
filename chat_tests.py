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

    module.AutoTokenizer = FakeTokenizer
    module.AutoModelForSeq2SeqLM = types.SimpleNamespace(from_pretrained=lambda name: FakeModel())
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

    def test_tfidf_context_returns_matching_document_indices(self):
        module = load_chat_module()

        class FakeVectorizer:
            def transform(self, questions):
                return questions

        module.docs = ["first document", "lend", "third document"]
        module.embedding_backend = {
            "mode": "tfidf",
            "vectorizer": FakeVectorizer(),
            "matrix": None,
        }
        with patch.object(
            module,
            "cosine_similarity",
            return_value=np.array([0.3, 0.9, 0.0]),
        ):
            context, indices = module._find_best_context("lend")

        self.assertEqual(context, "lend\nfirst document")
        self.assertEqual(indices, [1, 0])
        self.assertTrue(all(type(index) is int for index in indices))

    def test_listen_context_uses_user_text_alongside_speech_match(self):
        module = load_chat_module()
        module.listen_docs = module.load_text_files("docs_listen")

        context, _ = module._find_listen_context("late flight\nlate")

        self.assertIn("connection flight", context)
        self.assertEqual(module._find_listen_context("late")[0], "")

    def test_main_exits_immediately(self):
        module = load_chat_module()
        with patch("builtins.input", side_effect=["exit"]):
            with patch("builtins.print"):
                module.main()

    def test_chat_answer_uses_matching_context_when_model_is_unavailable(self):
        module = load_chat_module()
        module.qa_model = None
        module.qa_tokenizer = None

        self.assertEqual(module._chat_answer("lend", "borrow\nlend\nmoney"), "lend")

    def test_main_prints_grounded_answer_and_source(self):
        module = load_chat_module()
        module.docs = ["late"]
        module.listen_docs = ["connection flight"]
        module.banned_words = []

        printed = []
        with patch("builtins.input", side_effect=["late flight", "exit"]):
            with patch.object(
                module,
                "_find_best_context",
                return_value=("late", []),
            ), patch.object(
                module,
                "_chat_answer",
                return_value="connection flight",
            ) as chat_answer:
                with patch("builtins.print", side_effect=lambda *args: printed.append(args)):
                    module.main()

        output = "\n".join(" ".join(map(str, line)) for line in printed)
        chat_answer.assert_called_once_with("late flight", "connection flight")
        self.assertIn("Dopasowanie wejścia (docs_speech): late", output)
        self.assertIn("connection flight", output)
        self.assertIn("Źródło odpowiedzi: docs_listen.", output)
        self.assertIn("Dopasowanie pytania pochodzi z docs_speech.", output)
        self.assertNotIn("Pomysły_zewnętrzne", output)

    def test_chat_answer_reports_no_match_when_context_is_unrelated(self):
        module = load_chat_module()
        module.qa_model = None
        module.qa_tokenizer = None

        self.assertEqual(
            module._chat_answer("basic", "unrelated vocabulary"),
            "Nie znalazłem odpowiedniego kontekstu w dokumentach.",
        )


if __name__ == "__main__":
    unittest.main()
