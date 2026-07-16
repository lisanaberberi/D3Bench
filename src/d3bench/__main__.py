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

from pydantic import Field
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
            "When set, --datafile is ignored (the scenario picks its own dataset); "
            "--tools/--criteria override the scenario's [run] values if explicitly passed."
        ),
    )
    criteria: set[Criteria] = Field(
        default=set(["runtime", "cputime", "memory"]),
        description="Criteria to test. With --scenario, overrides the scenario's criteria if passed.",
    )
    tools: set[Framework] = Field(
        default=set(["Menelaus"]),
        description="List of tools to benchmark. With --scenario, overrides the scenario's tools if passed.",
    )
    datafile: Datafile = Field(
        default="energy",
        description="Dataset file name to use.",
    )
    output: Optional[str] = Field(
        default=None,
        description="File to save the results to. Defaults to 'results_<dataset>_<tools>_<timestamp>'.",
    )


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
        if "tools" in args.model_fields_set:
            logger.debug("Overriding scenario tools with CLI value: %s", args.tools)
            scenario.run.tools = list(args.tools)
        if "criteria" in args.model_fields_set:
            logger.debug("Overriding scenario criteria with CLI value: %s", args.criteria)
            scenario.run.criteria = list(args.criteria)
        dataset_slug = args.scenario.stem
        run_tools = scenario.run.tools
        results = scenario.run_benchmark()
    else:
        logger.info("Loading dataset and tools from d3bench")
        data = d3bench.DATASETS[args.datafile].split_data()
        tools = [d3bench.TOOLS[tool](data) for tool in args.tools]
        dataset_slug = args.datafile
        run_tools = list(args.tools)

        logger.info("Loading the benchmarks with the given criteria")
        logger.debug("Criteria: %s", args.criteria)
        results = d3bench.Results(tools, args.criteria)
    logger.debug("Results: %s", results)

    output = args.output
    if output is None:
        tools_slug = "_".join(sorted(tool.lower().replace("-", "") for tool in run_tools))
        timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        output = f"results_{dataset_slug}_{tools_slug}_{timestamp}"

    logger.info("Saving the results to the output file")
    logger.debug("Output file: %s", output)
    results.save_json(output=output)
    logger.info("Benchmark completed successfully")


# Run main function if the script is executed
if __name__ == "__main__":
    main(args=Arguments())
