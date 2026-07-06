import argparse
import json

import Dataset
import Tool
from Benchmark import Benchmark, Criteria

# Runs one tool's full benchmark in its own process. Spawned by Benchmark.runBenchmark() so that
# whatever memory the tool's detector libraries allocate is fully released back to the OS when
# this process exits, instead of accumulating across every tool run in one shared process.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to a JSON file describing the tool/dataset/criteria to run')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    tool_cls = getattr(Tool, cfg['tool_class'])
    tool = tool_cls(cfg['tool_name'], showReport=cfg['show_report'])

    dataset_cls = getattr(Dataset, cfg['dataset_class'])
    dataset = dataset_cls(cfg['dataset_path'])

    criterias = [Criteria[name] for name in cfg['criteria']]
    buildings = set(cfg['buildings'])

    benchmark = Benchmark(tool, dataset, criterias, buildings, cfg['vm'])
    benchmark._executeInProcess()

if __name__ == '__main__':
    main()
