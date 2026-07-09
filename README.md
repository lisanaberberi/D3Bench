# D3Bench: Benchmarking Open Source Tools for Dataset Drift Detection
=====================================================================

This repository contains the source code of the paper titled <a href=https://arxiv.org/abs/2404.18673>Open-Source Drift Detection Tools in Action: Insights from Two Use Cases</a>. The aim of D3Bench is to compare the most common tools for detecting dataset shifts. Through this comparison, deficiencies and limitations of the tools can be identified, and future research directions can be explored. Additionally, it can be determined which tools are mature enough for industrial use. The comparison criteria are as follows:

* **Functional Suitability**: The tool should reliably detect shifts.
* **Integration Capabilities**: The tool should be integrable into the MLOps pipeline, preferably with Grafana and MLflow.
* **Flexibility**: The tool should be able to handle various data types.
* **Usability**: The tool should be user-friendly and easy to use.
* **Runtime and Resource Usage**: The tool should perform calculations quickly and use as little memory as possible.

Tools currently benchmarked: **Alibi-Detect**, **Evidently AI**, **NannyML**, **Frouros**, **River**, and **Menelaus**.

## Setup

Clone the repository with submodules:
```shell
git clone https://git.sagresearch.de/kompaki/edd/d3-benchmark.git --recurse-submodules
```

Create a virtual environment and install the package:
```shell
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip setuptools

# Install d3bench together with all benchmarked drift detection tools
# (Evidently, NannyML, Alibi-Detect, Frouros, River, Menelaus)
pip3 install -e ".[full]"
```

If you only need the core package (e.g. to add a new tool or work on the
CLI) without pulling in every drift detection library, install it without
the `full` extra instead:
```shell
pip3 install -e .
```

## Structure

![](benchmark-highlevel.png)

* **src/d3bench**: The installable `d3bench` package containing the benchmarking process.
	* **config.py**: Defines the available frameworks, datasets, and evaluation criteria.
	* **datasets.py**: The `Dataset` classes bind a datafile and perform preprocessing, splitting the data into training and test sets.
	* **methods.py**: Taxonomy of the online/batch, concept/data drift detection methods implemented by each tool.
	* **tools/**: The `Tool` parent class and its subclasses — `Alibi-Detect`, `Evidently`, `NannyML`, `Frouros`, `River`, and `Menelaus`. Each tool implements the necessary preprocessing steps and the calculation of drift values for its methods.
	* **results.py** / **reports.py**: Run the benchmarks for the selected tools/criteria and collect the results.
	* **\_\_main\_\_.py**: Command-line entry point (see [Usage](#usage) below).
* **datafiles**: The datasets available for benchmarking (`energy`, `occupancy`).
* **scripts**: Shell scripts to run the benchmark repeatedly, e.g. `execute_benchmark_5_times.sh`.
* **results**: Benchmark results are stored here as JSON files, one per run. The Jupyter notebooks in this folder (and `tables.py` at the repo root) load and visualize these JSON files.

## Usage

Once installed, the benchmark is run as a module through its CLI:
```shell
python3 -m d3bench --tools Evidently NannyML --criteria runtime memory --datafile energy --output results/my_run
```

See all available options (tools, criteria, datafile, log level, etc.):
```shell
python3 -m d3bench --help
```
