"""
This module contains the configuration of the datasets and tools used in the
benchmarking process.
"""

import json
from collections.abc import Mapping
from typing import Generator

from pydantic.json import pydantic_encoder

from d3bench.benchmarks import get_reports
from d3bench.config import Criteria, Framework, MethodFamily, results_path
from d3bench.reports import Report
from d3bench.tools import Tool


class Results(Mapping):
    """Data class to store the results of the benchmark."""

    def __init__(self, tools: list[Tool], criteria: set[Criteria], families: frozenset[MethodFamily]):
        self.tools = {tool.name: tool for tool in tools}
        self.criteria = criteria
        self.families = families

    def __getitem__(self, index: Framework) -> list[Report]:
        generator = get_reports(tool=self.tools[index], criteria=self.criteria, families=self.families)
        return list(generator)

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Generator[list[Report], None, None]:
        return (self[tool] for tool in self.tools)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tools}, {self.criteria})"

    def save_json(self, output: str) -> None:
        """Save the results to a file."""
        with open(results_path / f"{output}.json", "w", encoding="utf-8") as f:
            json.dump(list(self), f, default=pydantic_encoder, indent=4)
