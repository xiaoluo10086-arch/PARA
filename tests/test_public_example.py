from pathlib import Path
import unittest

from para.reasoner import SUPPORTED, reason_query


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "data" / "example" / "can_call_class"


class PublicExampleTest(unittest.TestCase):
    def test_supported_query_has_grounded_proof(self) -> None:
        result = reason_query(
            task_dir=TASK,
            rule_library=TASK / "rule_library.json",
            query="canCallClass(order_service,payment_client)",
        )

        self.assertEqual(result["decision"], SUPPORTED)
        self.assertTrue(result["evidence"])
        self.assertEqual(
            set(result["evidence"][0]["body_facts"]),
            {
                "containsMethod(order_service,submit_order)",
                "callsMethod(submit_order,charge_payment)",
                "containsMethod(payment_client,charge_payment)",
            },
        )


if __name__ == "__main__":
    unittest.main()
