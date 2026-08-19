from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cfg import get_config
from pipeline.report import run
from tools.io import write_json


class ReportingTests(unittest.TestCase):
    def test_real_and_synthetic_results_are_written_to_separate_tables(self) -> None:
        with TemporaryDirectory() as directory:
            cfg = get_config()
            cfg["paths"]["project_root"] = directory
            cfg["run"]["task"] = "report_test"
            root = Path(directory) / "output" / "report_test"
            evaluation_path = (
                root / "metrics" / "rope" / "seed42" / "evaluation.json"
            )
            write_json(
                evaluation_path,
                {
                    "method": "rope",
                    "seed": 42,
                    "checkpoints": {
                        "pretrain": {
                    "length_degradation_rate": 0.1,
                    "data_sources": {
                        "natural_language": {
                            "fineweb_edu_held_out": {
                                "dataset": "fineweb",
                                "revision": "fixed",
                                "local_split": "test",
                                "sample_count": 10,
                                "program_generated": False,
                            },
                            "wikitext103": {
                                "dataset": "wikitext",
                                "revision": "fixed",
                                "local_split": "test",
                                "sample_count": 10,
                                "program_generated": False,
                            },
                        },
                        "real_long_document_qa": {
                            "qasper": {
                                "dataset": "qasper",
                                "revision": "fixed",
                                "split": "validation",
                                "leakage_check_complete": True,
                                "fineweb_documents_excluded_for_evaluation_overlap": 0,
                            }
                        },
                        "synthetic_control": {
                            "single_query": {
                                "generator": "controlled",
                                "generator_version": 1,
                                "samples_per_length": 2,
                                "program_generated": True,
                            }
                        },
                    },
                    "lengths": {
                        "64": {
                            "natural_language": {
                                "fineweb_edu_held_out_ppl": 2.0,
                                "wikitext103_ppl": 3.0,
                            },
                            "real_long_document_qa": {
                                "qasper": {
                                    "samples": 2,
                                    "answer_nll": 1.0,
                                    "token_f1": 0.5,
                                    "exact_match": 0.0,
                                    "evidence_utilization_gain": 0.2,
                                    "mean_evidence_distance_tokens": 40,
                                    "question_ids": ["q1", "q2"],
                                    "sample_sha256": ["h1", "h2"],
                                }
                            },
                            "synthetic_control": {
                                "single_query": {
                                    "accuracy": 0.5,
                                    "near_accuracy": 1.0,
                                    "far_accuracy": 0.0,
                                    "rcug": 0.1,
                                }
                            },
                        }
                    },
                        }
                    },
                },
            )
            result = run(cfg)
            natural_path = Path(result["tables"]["natural_language"])
            real_path = Path(result["tables"]["real_long_document_qa"])
            synthetic_path = Path(result["tables"]["synthetic_control"])
            with natural_path.open(encoding="utf-8-sig", newline="") as handle:
                natural = list(csv.DictReader(handle))
            with real_path.open(encoding="utf-8-sig", newline="") as handle:
                real = list(csv.DictReader(handle))
            with synthetic_path.open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                synthetic = list(csv.DictReader(handle))
            self.assertEqual(len(natural), 2)
            self.assertEqual(len(real), 1)
            self.assertEqual(len(synthetic), 1)
            self.assertEqual(real[0]["program_generated"], "False")
            self.assertEqual(synthetic[0]["program_generated"], "True")


if __name__ == "__main__":
    unittest.main()
