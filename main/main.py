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

def main():
    clean()
    ####################################
    # use can change or define the benchmark here:
    # 1. select dataset: True if you want to investigate energy dataset, False if you want to investigate Occupacy dataset
    energy = False 

    # 2. select the tools
    tools = {Evidently("Evidently", False), Evidently("Evidently", True),
              NannyML("NannyML", False), NannyML("NannyML", True),
              AlibiDetect("AlibiDetect", True),
              Frouros("Frouros", True),
              River("River", True)}

    # 3. select criteria
    criteria = [Criteria.FUNCTIONAL, Criteria.RUNTIME, Criteria.CPU_RUNTIME, Criteria.STORAGE]

    # 4. select if run on vm: True if run on vm, False if run locally
    vm = False

    # finished
    #####################################

    if energy:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'energy_data.csv')
        dataset= Data_Energy(path)

    else:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'occupacy_data.csv')
        dataset = Data_Occupacy(path)

    runBenchmark(buildings={1}, tests=criteria, tools=tools, vm = vm, dataset=dataset)
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
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'reports')
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