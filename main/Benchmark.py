from datetime import datetime
import time
from enum import Enum
import glob
import pandas as pd
from memory_profiler import memory_usage
import json
import os.path
import subprocess
import sys
import tempfile

class Criteria(Enum):
    FUNCTIONAL = 0
    RUNTIME = 1
    CPU_RUNTIME = 2
    STORAGE = 3

class Benchmark:
    # Class attributes
    runtime_avg  = 0
    runtime_max = 0
    runtime_cpu_avg  = 0
    runtime_cpu_max  = 0
    ram_avg = 0
    ram_max = 0

    def __init__(self, tool, dataset, criterias , buildings, vm):
        self.tool = tool
        self.dataset = dataset
        self.criterias = criterias
        self.buildings = buildings
        self.runOnVm = vm
        self.driftDetectionStats = {}
        self.tool.dataset_name = type(dataset).__name__.lower()

    # Runs the actual drift-detection work for this tool in its own short-lived subprocess
    # (via runner.py) instead of in this long-lived process. Tool.py imports tensorflow, torch,
    # evidently, nannyml, alibi-detect, frouros, and river all at module level, and running many
    # tools' full benchmarks back-to-back in one shared process was observed to spike memory to
    # 20-28GB (of 30GB total) and trigger an apparent kernel OOM-kill -- reproduced repeatedly even
    # after ruling out joblib/multiprocessing and cutting workload size. Isolating each tool in its
    # own process guarantees the OS reclaims whatever it allocated when that process exits,
    # regardless of the underlying cause.
    def runBenchmark(self):
        cfg = {
            'tool_class': type(self.tool).__name__,
            'tool_name': self.tool.name,
            'show_report': self.tool.showReport,
            'dataset_class': type(self.dataset).__name__,
            'dataset_path': self.dataset.path,
            'dataset_boundary': self.dataset.boundary,
            'criteria': [c.name for c in self.criterias],
            'buildings': list(self.buildings),
            'vm': self.runOnVm,
        }
        runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runner.py')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cfg, f)
            config_path = f.name
        try:
            # stdout/stderr are inherited (not captured), so the child's report printout still
            # shows up in the console exactly as if it ran in-process
            result = subprocess.run([sys.executable, runner_path, '--config', config_path])
            if result.returncode != 0:
                print(f"WARNING: benchmark subprocess for {self.tool.name} exited with code "
                      f"{result.returncode}; skipping to the next tool")
        finally:
            os.remove(config_path)

    # called by runner.py from inside the isolated subprocess
    def _executeInProcess(self):
        for criteria in self.criterias:
            if criteria == Criteria.FUNCTIONAL:
                self.runFunctional()
            elif criteria == Criteria.RUNTIME:
                self.runRuntime()
            elif criteria == Criteria.CPU_RUNTIME:
                self.runCPUruntime()
            elif criteria == Criteria.STORAGE:
                self.runStorage()

        # generate Report
        self.__printReport()

    def __printReport(self):
        self.driftDetectionStats=pd.DataFrame.from_dict(self.driftDetectionStats)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("==============================")
        name = self.tool.name
        if self.tool.showReport:
            name = name + " with report"
        print("Benchmark Report: {}".format(name))
        print("Report generated at: {}".format(current_time))
        print("==============================")
        for x in self.driftDetectionStats:
            print("Gebäude {}".format(x))
            building_stats = self.driftDetectionStats[x]
            column_names = self.tool.column_names
            
            # Extract the drift scores and is_drifted values
            for col in column_names:
                drift_score = {test: stats.get(f"{col}_drift_score", stats.get("drift_score"))
                                for test, stats in building_stats.items()
                                if isinstance(stats, dict)}
                is_drifted = {test: stats.get(f"{col}_is_drifted", stats.get("is_drifted"))
                                for test, stats in building_stats.items()
                                if isinstance(stats, dict)}
                print(f"Column: {col}")
                print(f"Drift Score: {drift_score}")
                print(f"Is Drifted: {is_drifted}")
                print()

            print()
        print("==============================")
        print("Runtime AVG: {:.8f} milliseconds".format(self.runtime_avg))
        print("Runtime MAX: {:.8f} milliseconds".format(self.runtime_max))
        print("CPU Runtime AVG: {:.7f} milliseconds".format(self.runtime_cpu_avg))
        print("CPU Runtime MAX: {:.7f} milliseconds".format(self.runtime_cpu_max))
        print("RAM Usage AVG: {:.7f} MiB".format(self.ram_avg))
        print("RAM Usage MAX: {:.7f} MiB".format(self.ram_max))
        print("==============================")

        self.__saveReport(current_time=current_time)

    
    def __saveReport(self, current_time):
        self.driftDetectionStats = pd.DataFrame.from_dict(self.driftDetectionStats)
        column_names = self.tool.column_names

        # Append data for each test to the report list
        report_data = []

        for x in self.driftDetectionStats:
            building_stats = self.driftDetectionStats[x]

            # Extract the test names
            tests = [test for test, stats in building_stats.items() if isinstance(stats, dict)]

            for test in tests:
                stats = building_stats[test]

                # Create a dictionary to hold the row's data
                row_data = {
                    'time': current_time,
                    'building': x,
                    'tool': self.tool.name,
                    'showReport': self.tool.showReport,
                    'test': test,
                    'runtime_avg': self.runtime_avg,
                    'runtime_max': self.runtime_max,
                    'cpu_runtime_avg': self.runtime_cpu_avg,
                    'cpu_runtime_max': self.runtime_cpu_max,
                    'ram_avg': self.ram_avg,
                    'ram_max': self.ram_max,
                    'run_on_vm': self.runOnVm,
                    # set only for tests that subsample x_ref/x_cur (e.g. AlibiDetect mmd/lsdd);
                    # NaN here means the test ran on the full ref/cur window like every other tool
                    'n_ref_used': stats.get('n_ref_used'),
                    'n_cur_used': stats.get('n_cur_used'),
                    'n_ref_total': stats.get('n_ref_total'),
                    'n_cur_total': stats.get('n_cur_total'),
                }

                for col in column_names:
                    col_drift_score = f"{col}_drift_score"
                    col_is_drifted = f"{col}_is_drifted"

                    drift_score = stats.get(col_drift_score)
                    is_drifted = stats.get(col_is_drifted)

                    if (drift_score == None) & (is_drifted == None):
                        row_data['all_col_drift_score'] = stats.get('drift_score')
                        row_data['all_col_is_drifted'] = stats.get('is_drifted')
                    else:
                        row_data[col_drift_score] = drift_score
                        row_data[col_is_drifted] = is_drifted

                report_data.append(row_data)

        # Create a DataFrame from the report data
        report_df = pd.DataFrame(report_data)

        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        has_perf_criteria = any(c in self.criterias for c in (Criteria.RUNTIME, Criteria.CPU_RUNTIME, Criteria.STORAGE))

        if has_perf_criteria:
            # one file per execution: runtime/cpu/ram numbers are only comparable within a single run, not across appended rows
            out_dir = os.path.join(results_dir, 'non-functional')
            os.makedirs(out_dir, exist_ok=True)
            execution = 1
            while os.path.exists(os.path.join(out_dir, f'benchmark_report_{self.tool.dataset_name}_execution{execution}.csv')):
                execution += 1
            report_df.to_csv(os.path.join(out_dir, f'benchmark_report_{self.tool.dataset_name}_execution{execution}.csv'), index=False)
        else:
            out_dir = os.path.join(results_dir, 'functional', self.tool.dataset_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'benchmark_report.csv')
            if os.path.exists(out_path):
                report_df.to_csv(out_path, mode='a', index=False, header=False)
            else:
                report_df.to_csv(out_path, index=False)

    def runFunctional(self):
        for building_id in self.buildings:
            # Split into reference and current dataset
            ref, cur = self.dataset.splitTrainTest(building_id)
            self.driftDetectionStats[building_id] = {}

            # runDriftDetection without report generation
            my_dict = self.tool.runDriftdetection(ref, cur, building_id)
            self.ref = ref
            if not my_dict:
                print("Dict from building {} is empty" .format(building_id))
            else:
                self.driftDetectionStats[building_id].update(my_dict)

    # measures elapsed time using wall-clock time (include: waiting time for resources, dependant on other processes): time in ms
    def runRuntime(self):  
        runtime_sum = 0  
        self.runtime_max = 0

        for building_id in self.buildings:
            # Split into reference and current dataset
            ref, cur = self.dataset.splitTrainTest(building_id)

            # Timestamp before executing drift detection
            st = time.time()
            self.tool.runDriftdetection(ref, cur, building_id)

            # Timestamp after executing, compute runtime in ms, divided by count of algorithms of tool, result: avg runtime of 1 algorithm for 1 building
            runtime_result = ((time.time() - st)/len(self.tool.methods))  * 1000 
            runtime_sum += runtime_result

            # set maximal runtime
            self.runtime_max = max(self.runtime_max, runtime_result)

        # compute average runtime
        self.runtime_avg = runtime_sum / len(self.buildings) 

    # measures cpu resources (user and system) consumed by the process (exclude: waiting time for resources): time in ms
    def runCPUruntime(self):
        cpu_sum = 0  
        self.runtime_cpu_max = 0
        for building_id in self.buildings:
            ref, cur = self.dataset.splitTrainTest(building_id)

            # Start measuring CPU Usage (include user and system cpu time)
            st = time.process_time() 
            self.tool.runDriftdetection(ref, cur, building_id)

            # End measuring CPU Usage, compute cpu in GB
            end = time.process_time()
            cpu_result = ((end-st) / len(self.tool.methods)) * 1000
            cpu_sum += cpu_result

            # set maximal cpu
            self.runtime_cpu_max = max(self.runtime_cpu_max, cpu_result)

        # compute average cpu
        self.runtime_cpu_avg = cpu_sum / len(self.buildings) 

    def runStorage(self):
        self.ram_max = 0
        mem = []
        for building_id in self.buildings:
            ref, cur = self.dataset.splitTrainTest(building_id)
            mem = mem + memory_usage((self.tool.runDriftdetection, (ref,cur,building_id)))

        # compute average storage
        self.ram_avg = sum(mem) / len(mem)
        self.ram_max = max(mem)

# Combines every per-tool report for a dataset (the many results/non-functional/*execution*.csv
# files, one per tool instance, plus results/functional/<dataset>/benchmark_report.csv if it
# exists) into a single results/summary_<dataset>.csv. Columns are the union of whatever each
# source has -- NaN where a column doesn't apply -- so nothing is collapsed/aggregated;
def summarize(dataset_name):
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    frames = []

    non_functional_pattern = os.path.join(results_dir, 'non-functional', f'benchmark_report_{dataset_name}_execution*.csv')
    for file_name in sorted(glob.glob(non_functional_pattern)):
        df = pd.read_csv(file_name)
        if not df.empty:
            frames.append(df)

    functional_path = os.path.join(results_dir, 'functional', dataset_name, 'benchmark_report.csv')
    if os.path.exists(functional_path):
        df = pd.read_csv(functional_path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return None

    summary_df = pd.concat(frames, ignore_index=True, sort=False)
    summary_df = summary_df.sort_values(['tool', 'showReport', 'test']).reset_index(drop=True)

    out_path = os.path.join(results_dir, f'summary_{dataset_name}.csv')
    summary_df.to_csv(out_path, index=False)
    return out_path