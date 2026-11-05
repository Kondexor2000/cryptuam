import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, mock_open, patch

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
        module.listen_docs = ["connection flight\nlate"]

        context, _ = module._find_listen_context("late flight\nlate")

        self.assertIn("connection flight", context)
        self.assertEqual(module._find_listen_context("late")[0], "late")

    def test_main_exits_immediately(self):
        module = load_chat_module()
        with patch("builtins.input", side_effect=["exit"]):
            with patch("builtins.print"):
                module.main()

    def test_chat_answer_uses_ollama_response_and_vocabulary_context(self):
        module = load_chat_module()
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(
            {"message": {"content": "I can lend you a book."}}
        ).encode("utf-8")

        with patch.object(
            module.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            answer = module._chat_answer("lend", "borrow\nlend\nmoney")

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(answer, "I can lend you a book.")
        self.assertIn("borrow\nlend\nmoney", payload["messages"][-1]["content"])
        self.assertIn("Always answer in English", payload["messages"][0]["content"])
        self.assertIn("Do not invent definitions", payload["messages"][0]["content"])

    def test_main_passes_speech_and_listen_context_to_chat_answer(self):
        module = load_chat_module()
        module.docs = ["late"]
        module.speech_docs = ["late flight"]
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
        chat_answer.assert_called_once_with(
            "late flight",
            "docs_speech:\nlate\ndocs_listen:\nconnection flight",
            [],
        )
        self.assertIn("Asystent: connection flight", output)

    def test_main_rejects_question_without_docs_speech_word(self):
        module = load_chat_module()
        module.speech_docs = ["connection flight"]
        module.banned_words = []

        printed = []
        with patch("builtins.input", side_effect=["basic", "exit"]):
            with patch.object(module, "_chat_answer") as chat_answer:
                with patch("builtins.print", side_effect=lambda *args: printed.append(args)):
                    module.main()

        chat_answer.assert_not_called()
        output = "\n".join(" ".join(map(str, line)) for line in printed)
        self.assertIn("Please try again using at least one word", output)

    def test_chat_answer_reports_ollama_unavailable(self):
        module = load_chat_module()

        with patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=module.urllib.error.URLError("not running"),
        ):
            answer = module._chat_answer("lend", "borrow\nlend\nmoney")

        self.assertIn("Nie mogę połączyć się z Ollama", answer)

    def test_chat_answer_disables_model_thinking(self):
        module = load_chat_module()
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(
            {"message": {"content": "Cześć!"}}
        ).encode("utf-8")

        with patch.object(
            module.urllib.request, "urlopen", return_value=response
        ) as urlopen:
            answer = module._chat_answer("good morning", "")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertFalse(payload["think"])
        self.assertEqual(answer, "Cześć!")


if __name__ == "__main__":
    unittest.main()
