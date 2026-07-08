import argparse
import Benchmark
from Benchmark import Criteria
from Tool import Evidently
from Tool import AlibiDetect
from Tool import NannyML
from Tool import Frouros
from Tool import River
import Dataset
from Dataset import Data_Occupacy
from Dataset import Data_Energy
import os
import shutil
import tomli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_REGISTRY = {'energy': Data_Energy, 'occupancy': Data_Occupacy}
TOOL_REGISTRY = {'Evidently': Evidently, 'NannyML': NannyML, 'AlibiDetect': AlibiDetect,
                  'Frouros': Frouros, 'River': River}

# Loads a scenario .toml (see ../scenarios/) into the same (dataset, tools, criteria, buildings,
# vm) shape that runBenchmark() already takes, so a scenario file is a declarative, reproducible
# stand-in for the "user can change or define the benchmark here" block below.
def load_scenario(scenario_path):
    with open(scenario_path, 'rb') as f:
        cfg = tomli.load(f)

    data_cfg = cfg['data']
    run_cfg = cfg['run']

    dataset_cls = DATASET_REGISTRY[data_cfg['dataset']]
    path = os.path.join(REPO_ROOT, data_cfg['path'])
    dataset = dataset_cls(path, boundary=data_cfg.get('split_boundary'))

    show_report = run_cfg.get('show_report', True)
    tools = {TOOL_REGISTRY[name](name, show_report) for name in run_cfg['tools']}
    criteria = [Criteria[name] for name in run_cfg['criteria']]
    buildings = {data_cfg['building_id']}
    vm = run_cfg.get('vm', False)

    return dataset, tools, criteria, buildings, vm

def main():
    #clean()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='Path to a scenario .toml file (see ../scenarios/). '
                         'If omitted, falls back to the hardcoded defaults below.')
    args = parser.parse_args()

    if args.scenario:
        dataset, tools, criteria, buildings, vm = load_scenario(args.scenario)
    else:
        ####################################
        # use can change or define the benchmark here:
        # 1. select dataset: True if you want to investigate energy dataset, False if you want to investigate Occupacy dataset
        energy = False

        # 2. select the tools
        tools = {Evidently("Evidently", True),
                  NannyML("NannyML", True),
                  AlibiDetect("AlibiDetect", True),
                  Frouros("Frouros", True), #memory peak issue
                  River("River", True)}

        # 3. select criteria
        # NOTE: Criteria.STORAGE is left out by default. It measures RAM via memory_profiler's
        # memory_usage(), which forks this process (already carrying tensorflow/torch/alibi-detect/
        # evidently/nannyml/frouros/river from Tool.py's imports) to sample the child's RSS. With
        # Frouros in `tools`, this reproducibly spiked system memory to 20-28GB (of 30GB total) within
        # seconds to minutes -- confirmed 3x, including one apparent kernel OOM-kill
        criteria = [Criteria.FUNCTIONAL, Criteria.RUNTIME, Criteria.CPU_RUNTIME]

        # 4. select if run on vm: True if run on vm, False if run locally
        vm = False

        # finished
        #####################################

        if energy:
            path = os.path.join(REPO_ROOT, 'data', 'energy_data.csv')
            dataset = Data_Energy(path)
        else:
            path = os.path.join(REPO_ROOT, 'data', 'occupacy_data.csv')
            dataset = Data_Occupacy(path)

        buildings = {1}

    runBenchmark(buildings=buildings, tests=criteria, tools=tools, vm=vm, dataset=dataset)

    dataset_name = type(dataset).__name__.lower()
    summary_path = Benchmark.summarize(dataset_name)
    if summary_path:
        print(f"Summary written to {summary_path}")

    print("---------Benchmark execution finished---------")

# one benchmark execution with given criteria, tools and dataset
def runBenchmark(buildings = {1}, tests=[Criteria.FUNCTIONAL, Criteria.RUNTIME, Criteria.CPU_RUNTIME, Criteria.STORAGE],
                  tools={(Evidently("Evidently", showReport=False))}, vm = False, 
                  dataset=Data_Energy(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'energy_data.csv'))):
    for tool in tools:
        benchmark = Benchmark.Benchmark(tool, dataset, tests, buildings, vm)
        benchmark.runBenchmark()

# delete all reports
def clean():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    reports_dir = os.path.join(results_dir, 'reports')

    # non-functional/ and functional/ accumulate one CSV per Benchmark.runBenchmark() call with no
    # cap, so leftover files from a previous (or overlapping) run stay mixed in with the current
    # one and the execution numbering no longer means what it implies -- clear both every run
    for sub in ('non-functional', 'functional'):
        shutil.rmtree(os.path.join(results_dir, sub), ignore_errors=True)
    evidently_tests = {'kolmogorov_smirnov', 'anderson', 'cramer_von_mises', 'ed', 'es', 'hellinger',
             'jensenshannon', 'kl_div', 'mannw', 'psi', 't_test', 'wasserstein'}
    nannyml_tests = {'kolmogorov_smirnov', 'wasserstein', 'jensen_shannon', 'hellinger'}
    alibidetect_tests = {'kolmogorov_smirnov', 'cramer_von_mises', 'spotdiff', 'mmd', 'lsdd'}
    frouros_tests = {'kolmogorov_smirnov', 'cramer_von_mises', 'anderson', 'mannw', 't_test', 'chi_square',
             'bws', 'kuiper', 'psi', 'kl_div', 'jensenshannon', 'hellinger', 'ed', 'bhattacharyya', 'hi', 'mmd'}
    river_tests = {'kswin', 'adwin', 'page_hinkley'}

    for dataset_name in ('data_energy', 'data_occupacy'):
        for i in range(1, 37):
            for test in evidently_tests:
                file_name = os.path.join(reports_dir, dataset_name, 'evidently', "evidently_report_{}_{}.html".format(i, test))
                if os.path.exists(file_name):
                    os.remove(file_name)
            for test in nannyml_tests:
                for kind in ('dist', 'drift'):
                    file_name = os.path.join(reports_dir, dataset_name, 'nannyml', "nannyml_report_{}_{}_{}.svg".format(kind, i, test))
                    if os.path.exists(file_name):
                        os.remove(file_name)
            for test in alibidetect_tests:
                file_name = os.path.join(reports_dir, dataset_name, 'alibidetect', "alibidetect_report_{}_{}.svg".format(i, test))
                if os.path.exists(file_name):
                    os.remove(file_name)
            for test in frouros_tests:
                file_name = os.path.join(reports_dir, dataset_name, 'frouros', "frouros_report_{}_{}.svg".format(i, test))
                if os.path.exists(file_name):
                    os.remove(file_name)
            for test in river_tests:
                file_name = os.path.join(reports_dir, dataset_name, 'river', "river_report_{}_{}.svg".format(i, test))
                if os.path.exists(file_name):
                    os.remove(file_name)

if __name__ == "__main__":
    main()