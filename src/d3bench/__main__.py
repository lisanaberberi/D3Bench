"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import datetime as dt
import logging
from pathlib import Path
from typing import Literal, Optional, TypeAlias

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict
from rich.logging import RichHandler

import d3bench
from d3bench import Criteria, Datafile, Framework
from d3bench.scenario import Scenario
from d3bench.utils import BaseArguments

# pylint: disable=too-few-public-methods


LogLevel: TypeAlias = Literal["debug", "info", "warning", "error", "critical"]
logger = logging.getLogger(__name__)


class Arguments(BaseArguments):
    """
    This module provides a command-line interface to run D3Bench benchmarks.

    The script allows users to specify various parameters for the benchmark,
    including the buildings to benchmark, the criteria to test, the tools to
    use, whether to run on a VM, the dataset to use, and the logging level.
    """  # Description for the script help message

    # Class attributes
    model_config = SettingsConfigDict(
        cli_prog_name=f"python -m {__package__}",
    )

    # Logging and reporting
    log_level: LogLevel = Field(
        default="info",
        description="Logging level.",
    )
    scenario: Optional[Path] = Field(
        default=None,
        description=(
            "Path to a scenario TOML file (e.g. scenarios/energy_covariate.toml). "
            "When set, this takes precedence over --datafile/--tools/--criteria."
        ),
    )
    criteria: set[Criteria] = Field(
        default=set(["runtime", "cputime", "memory"]),
        description="Criteria to test.",
    )
    tools: set[Framework] = Field(
        default=set(["Menelaus"]),
        description="List of tools to benchmark.",
    )
    datafile: Datafile = Field(
        default="energy",
        description="Dataset file name to use.",
    )
    output: Optional[str] = Field(
        default=None,
        description="File to save the results to. Defaults to 'results_<tools>_<timestamp>'.",
    )

    @model_validator(mode="after")
    def set_default_output(self) -> "Arguments":
        """Derive the output file name from the scenario/tools if not set explicitly."""
        if self.output is None:
            timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            if self.scenario is not None:
                slug = self.scenario.stem
            else:
                slug = "_".join(sorted(tool.lower().replace("-", "") for tool in self.tools))
            self.output = f"results_{slug}_{timestamp}"
        return self


def main(args: Arguments) -> None:
    """Run the benchmark with the given arguments."""

    # Set the logging level from the arguments
    logging.basicConfig(
        handlers=[RichHandler(rich_tracebacks=True)],
        level=args.log_level.upper(),
        force=True,
    )
    logger.debug("Call arguments: %s", args)

    if args.scenario is not None:
        logger.info("Loading scenario from %s", args.scenario)
        scenario = Scenario.from_toml(args.scenario)
        results = scenario.run_benchmark()
    else:
        logger.info("Loading dataset and tools from d3bench")
        data = d3bench.DATASETS[args.datafile].split_data()
        tools = [d3bench.TOOLS[tool](data) for tool in args.tools]

        logger.info("Loading the benchmarks with the given criteria")
        logger.debug("Criteria: %s", args.criteria)
        results = d3bench.Results(tools, args.criteria)
    logger.debug("Results: %s", results)

    logger.info("Saving the results to the output file")
    logger.debug("Output file: %s", args.output)
    results.save_json(output=args.output)
    logger.info("Benchmark completed successfully")


# Run main function if the script is executed
if __name__ == "__main__":
    main(args=Arguments())
