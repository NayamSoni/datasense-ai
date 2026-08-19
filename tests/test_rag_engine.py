import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent_agent import detect_intent
from rag_engine import (
    available_starter_industries,
    build_knowledge_index,
    chunk_documents,
    format_retrieved_context,
    retrieve_knowledge,
    starter_glossary_document,
)


def fake_embeddings(texts, model):
    """Small deterministic embedding used only to test retrieval behaviour."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append([
            1.0 if "allocation" in lowered else 0.0,
            1.0 if "cancellation" in lowered else 0.0,
            0.2,
        ])
    return vectors


class RagEngineTests(unittest.TestCase):
    def test_starter_glossary_has_ten_isolated_industries(self):
        industries = available_starter_industries()

        self.assertEqual(len(industries), 10)
        self.assertIn("Hospitality", industries)
        self.assertIn("SaaS & Product Analytics", industries)

        name, document = starter_glossary_document("Hospitality")
        chunks = chunk_documents([(name, document)])

        self.assertEqual(len(chunks), 5)
        self.assertTrue(all("industry: Hospitality" in item["text"] for item in chunks))
        self.assertFalse(any("industry: Marketing" in item["text"] for item in chunks))

    def test_csv_rows_keep_source_locations(self):
        glossary = (
            b"term,definition,formula\n"
            b"Allocation Rate,Share of leads allocated,allocations / unique leads\n"
            b"Cancellation Rate,Share of bookings cancelled,cancelled / bookings\n"
        )

        chunks = chunk_documents([("kpi_glossary.csv", glossary)])

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["source"], "kpi_glossary.csv")
        self.assertEqual(chunks[0]["location"], "row 2")
        self.assertIn("allocations / unique leads", chunks[0]["text"])

    def test_semantic_retrieval_returns_best_source(self):
        documents = [
            (
                "allocation.md",
                b"Allocation rate is allocations divided by unique leads.",
            ),
            (
                "cancellations.md",
                b"Cancellation rate is cancelled bookings divided by bookings.",
            ),
        ]
        index = build_knowledge_index(
            documents,
            model="test-model",
            embedding_function=fake_embeddings,
        )

        matches = retrieve_knowledge(
            "What is the allocation formula?",
            index,
            top_k=1,
            embedding_function=fake_embeddings,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["source"], "allocation.md")
        self.assertGreater(matches[0]["score"], 0.9)
        self.assertIn("[Source 1", format_retrieved_context(matches))

    def test_knowledge_questions_have_their_own_intent(self):
        self.assertEqual(
            detect_intent("What is the definition of allocation rate?"),
            "KNOWLEDGE",
        )
        self.assertEqual(
            detect_intent("According to the policy, when is a lead eligible?"),
            "KNOWLEDGE",
        )
        self.assertEqual(
            detect_intent("What does allocation rate mean?"),
            "KNOWLEDGE",
        )
        self.assertEqual(
            detect_intent("What is allocation rate?", knowledge_available=True),
            "KNOWLEDGE",
        )
        self.assertEqual(
            detect_intent("What is total revenue by city?", knowledge_available=True),
            "CALCULATE",
        )


if __name__ == "__main__":
    unittest.main()
