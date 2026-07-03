from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import *
from evidently.legacy.pipeline.column_mapping import ColumnMapping

import nannyml as nml # pip install nannyml

from alibi_detect.cd import KSDrift, CVMDrift, SpotTheDiffDrift, MMDDrift, LSDDDrift

from enum import Enum
import os
import pandas as pd # pip install pandas
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class METHODS(Enum):
    KOLMOGOROV_SMIRNOV = 0 # K-S Test
    WASSERSTEIN = 1 # Wasserstein Distance Normed
    KLD = 2 # Kullback-Leibler divergence
    PSI = 3 # Population Stability Index
    JSD = 4 # Jenson-Shannon Distance
    AD = 5 # Anderson-Darling
    CVM = 6 # Cramer-von-Mises
    HD = 7 # Hellinger distance
    MWURT = 8 # Mann-Whitney U-Rank Test
    ED = 9 # Energy-distance
    ES = 10 # Epps-Singleton
    TT = 11 # T-Test
    SPOTDIFF = 12 # Spot-The-Difference Test
    MMD = 13 # Maximum Mean Discrepancy
    LSDD = 14 # Least-Squares Density Difference

class Tool:
    # Class attributes
    name = "Tool"

    def __init__(self, name):
        self.name = name
        self.showReport = False
        self.dataset_name = "default"

    def preprocess(self):
        pass

    def runDriftdetection(self, ref, cur, building_id):
        pass

    def __runDriftdetectiontest(self, building_id, test):
        pass

    # anchors report output to main/results/reports/<dataset>/<tool>/ regardless of the CWD the script is run from
    def _report_path(self, tool_folder, file_name):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'reports', self.dataset_name, tool_folder)
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, file_name)

class Evidently(Tool):
    def __init__(self, name, showReport=False):
        super().__init__(name)
        self.showReport = showReport
        #self.methods = {}
        self.methods = {METHODS.WASSERSTEIN, METHODS.KLD, METHODS.PSI, METHODS.JSD, METHODS.AD, METHODS.CVM, METHODS.HD, 
               METHODS.MWURT, METHODS.ED, METHODS.ES, METHODS.TT, METHODS.KOLMOGOROV_SMIRNOV}

    def preprocess(self):
        if 'consumption' in self.ref:
            self.ref.rename(columns={'consumption': 'target'}, inplace=True)
            self.cur.rename(columns={'consumption': 'target'}, inplace=True)
            self.ref = self.ref.drop(columns={'ids'}).reset_index(drop=True)
            self.cur = self.cur.drop(columns={'ids'}).reset_index(drop=True)
            self.ref['target'] = pd.to_numeric(self.ref['target'])
            self.ref['temp_outside'] = pd.to_numeric(self.ref['temp_outside'])
        if 'prob_predicted' in self.ref:
            self.ref = self.ref.drop(columns={'prob_predicted', 'predicted'})
            self.cur = self.cur.drop(columns={'prob_predicted', 'predicted'})
        self.column_names = list(self.ref.columns)

        column_mapping = ColumnMapping()
        column_mapping.datetime = 'time'

    #@profile
    def runDriftdetection(self, ref, cur, building_id):
        self.ref = ref
        self.cur = cur
        self.preprocess()
        my_dict = {}
        for test in self.methods:
            if test == METHODS.WASSERSTEIN:
                my_dict['Wasserstein Distanz'] = self.__runDriftdetectiontest(building_id, 'wasserstein')
            elif test == METHODS.KLD:
                my_dict['K-L Divergence'] = self.__runDriftdetectiontest(building_id, 'kl_div')
            elif test == METHODS.PSI:
                my_dict['PSI'] = self.__runDriftdetectiontest(building_id, 'psi')
            elif test == METHODS.JSD:
                my_dict['J-S Distance'] = self.__runDriftdetectiontest(building_id, 'jensenshannon')
            elif test == METHODS.AD:
                my_dict['Anderson-Darling'] = self.__runDriftdetectiontest(building_id, 'anderson')
            elif test == METHODS.CVM:
                my_dict['Cramer-von-Mises'] = self.__runDriftdetectiontest(building_id, 'cramer_von_mises')
            elif test == METHODS.HD:
                my_dict['Hellinger-Distance'] = self.__runDriftdetectiontest(building_id, 'hellinger')
            elif test == METHODS.MWURT:
                my_dict['Mann-Whitney U-Rank Test'] = self.__runDriftdetectiontest(building_id, 'mannw')
            elif test == METHODS.ED:
                my_dict['Energy-Distance'] = self.__runDriftdetectiontest(building_id, 'ed')
            elif test == METHODS.ES:
                try:
                    my_dict['Epps-Singleton'] = self.__runDriftdetectiontest(building_id, 'es')
                except:
                    my_dict['Epps-Singleton'] = 'no result'
            elif test == METHODS.TT:
                my_dict['T-Test'] = self.__runDriftdetectiontest(building_id, 't_test')
            elif test == METHODS.KOLMOGOROV_SMIRNOV:
                my_dict['K-S Test'] = self.__runDriftdetectiontest(building_id, 'ks')

        return my_dict

    #@profile
    def __runDriftdetectiontest(self, building_id, test):
        # create TargetDriftReport in dict
        report = Report(metrics=[
            DataDriftPreset(stattest=test)
        ])
        report.run(reference_data=self.ref, current_data=self.cur)
        if(self.showReport):
            file_name = self._report_path('evidently', "evidently_report_{}_{}.html".format(building_id, test))
            report.save_html(file_name)
        report_dict = report.as_dict()

        # add into dictionary
        my_dict = {}
        for col in self.column_names:
            my_dict[f"{col}_drift_score"] = report_dict['metrics'][1]['result']['drift_by_columns'][col]['drift_score']
            my_dict[f"{col}_is_drifted"] = report_dict['metrics'][1]['result']['drift_by_columns'][col]['drift_detected']

        return my_dict
    
class NannyML(Tool):
    def __init__(self, name, showReport=False):
        super().__init__(name)
        self.showReport = showReport
        self.methods = {METHODS.KOLMOGOROV_SMIRNOV, METHODS.WASSERSTEIN, METHODS.JSD, METHODS.HD}

    def preprocess(self):
        if 'temp_outside' in self.ref:
            self.ref = self.ref.drop(columns={'ids'})
            self.cur = self.cur.drop(columns={'ids'})
        if 'prob_predicted' in self.ref:
            self.ref = self.ref.drop(columns={'prob_predicted', 'predicted'})
            self.cur = self.cur.drop(columns={'prob_predicted', 'predicted'})
        self.ref['time'] = self.ref.index
        self.ref = self.ref.reset_index(drop=True)
        self.cur['time'] = self.cur.index
        self.cur = self.cur.reset_index(drop=True)
        self.column_names = [col for col in self.ref.columns if col != 'time']
    
    #@profile
    def runDriftdetection(self, ref, cur, building_id):
        self.ref = ref
        self.cur = cur
        self.preprocess()

        my_dict = {}
        for test in self.methods:
            if test == METHODS.KOLMOGOROV_SMIRNOV:
                my_dict['K-S Test'] = self.__runDriftdetectiontest(building_id, 'kolmogorov_smirnov')
            elif test == METHODS.WASSERSTEIN:
                my_dict['Wasserstein Distance'] = self.__runDriftdetectiontest(building_id, 'wasserstein')
            elif test == METHODS.JSD:
                my_dict['J-S Distance'] = self.__runDriftdetectiontest(building_id, 'jensen_shannon')
            elif test == METHODS.HD:
                my_dict['Hellinger-Distance'] = self.__runDriftdetectiontest(building_id, 'hellinger')

        return my_dict
   
    # calculates drift score based on chunks
    # drift score is the mean of all chunks
    # is_drifted is the mean of True/False depending on the threshold, computed to %
    #@profile
    def __runDriftdetectiontest(self, building_id, test):
        calc = nml.UnivariateDriftCalculator(
            column_names=self.column_names,
            timestamp_column_name='time',
            continuous_methods=[test],
            thresholds = {
                'kolmogorov_smirnov': nml.thresholds.StandardDeviationThreshold(std_lower_multiplier=None),
                'jensen_shannon':  nml.thresholds.ConstantThreshold(upper=0.1),
                'wasserstein':  nml.thresholds.StandardDeviationThreshold(std_lower_multiplier=None),
                'hellinger':  nml.thresholds.ConstantThreshold(upper=0.1),}
        )

        calc.fit(self.ref)
        results = calc.calculate(self.cur)
        df = results.filter(period='analysis', column_names=self.column_names).to_df()

        if(self.showReport):
            figure = results.filter(column_names=results.continuous_column_names, methods=[test]).plot(kind='distribution')
            figure.write_image(self._report_path('nannyml', f'nannyml_report_dist_{building_id}_{test}.svg'))
            figure = results.filter(column_names=results.continuous_column_names, methods=[test]).plot(kind='drift')
            figure.write_image(self._report_path('nannyml', f'nannyml_report_drift_{building_id}_{test}.svg'))
            
        # add into dictionary
        my_dict = {}
        for col in self.column_names:
            my_dict[f"{col}_drift_score"] = df[col][test]['value'].mean()
            my_dict[f"{col}_is_drifted"] = str(round(df[col][test]['alert'].mean() * 100, 1)) + " % drifted"

        return my_dict

class AlibiDetect(Tool):
    # MMD/LSDD build an O(n^2) kernel matrix; cap sample size so it doesn't OOM on large windows
    MAX_KERNEL_SAMPLES = 2000

    def __init__(self, name, showReport=False):
        super().__init__(name)
        self.showReport = showReport
        self.methods = {METHODS.KOLMOGOROV_SMIRNOV, METHODS.CVM, METHODS.SPOTDIFF, METHODS.MMD, METHODS.LSDD}

    def __subsample(self, data, max_samples=MAX_KERNEL_SAMPLES):
        if len(data) <= max_samples:
            return data
        rng = np.random.default_rng(0)
        idx = rng.choice(len(data), size=max_samples, replace=False)
        return data[idx]

    def preprocess(self):
        if 'prob_predicted' in self.ref:
            self.ref = self.ref.drop(columns={'predicted', 'prob_predicted'})
            self.cur = self.cur.drop(columns={'predicted', 'prob_predicted'})
        if 'consumption' in self.ref:
            self.ref = self.ref.drop(columns={'ids'})
            self.cur = self.cur.drop(columns={'ids'})
        self.column_names = list(self.cur)
        self.ref = self.ref.to_numpy()
        self.cur = self.cur.to_numpy()

    def runDriftdetection(self, ref, cur, building_id):
        self.ref = ref
        self.cur = cur
        self.preprocess()

        my_dict = {}
        for test in self.methods:
            if test == METHODS.KOLMOGOROV_SMIRNOV:
                my_dict['K-S Test'] = self.__runDriftdetectiontest(building_id, 'kolmogorov_smirnov')
            elif test == METHODS.CVM:
                my_dict['Cramer-von-Mises'] = self.__runDriftdetectiontest(building_id, 'cramer_von_mises')
            elif test == METHODS.SPOTDIFF:
                my_dict['Spot-the-diff'] = self.__runDriftdetectiontest(building_id, 'spotdiff')
            elif test == METHODS.MMD:
                my_dict['Maximum Mean Discrepancy'] = self.__runDriftdetectiontest(building_id, 'mmd')
            elif test == METHODS.LSDD:
                my_dict['Least-Squares Density Difference'] = self.__runDriftdetectiontest(building_id, 'lsdd')

        return my_dict

    def __runDriftdetectiontest(self, building_id, test):
        report_dict = {}
        my_dict = {}

        if test == 'kolmogorov_smirnov':
            cd = KSDrift(x_ref = self.ref)
            report_dict = cd.predict(self.cur, drift_type='feature', return_p_val=True)
            for i in range(len(self.column_names)):
                col = self.column_names[i]
                my_dict[f"{col}_drift_score"] = report_dict['data']['p_val'][i]
                my_dict[f"{col}_is_drifted"] = report_dict['data']['is_drift'][i]
            if self.showReport:
                self.__saveDistributionReport(building_id, test, report_dict['data']['p_val'], report_dict['data']['is_drift'])
        elif test == 'cramer_von_mises':
            cd = CVMDrift(x_ref = self.ref)
            report_dict = cd.predict(self.cur, drift_type='feature', return_p_val=True)
            for i in range(len(self.column_names)):
                col = self.column_names[i]
                my_dict[f"{col}_drift_score"] = report_dict['data']['p_val'][i]
                my_dict[f"{col}_is_drifted"] = report_dict['data']['is_drift'][i]
            if self.showReport:
                self.__saveDistributionReport(building_id, test, report_dict['data']['p_val'], report_dict['data']['is_drift'])
        elif test == 'spotdiff':
            self.ref, self.cur = np.asarray(self.ref, np.float32), np.asarray(self.cur, np.float32)
            cd = SpotTheDiffDrift(x_ref = self.ref)
            report_dict = cd.predict(self.cur, return_p_val=True)
            score = report_dict['data']['p_val']
            drifted = report_dict['data']['is_drift']
            my_dict = {
                'drift_score': score,
                'is_drifted': drifted
            }
            if self.showReport:
                self.__saveGlobalReport(building_id, test, score, drifted)
        elif test == 'mmd':
            cd = MMDDrift(x_ref = self.__subsample(self.ref))
            report_dict = cd.predict(self.__subsample(self.cur), return_p_val=True)
            score = report_dict['data']['p_val']
            drifted = report_dict['data']['is_drift']
            my_dict = {
                'drift_score': score,
                'is_drifted': drifted
            }
            if self.showReport:
                self.__saveGlobalReport(building_id, test, score, drifted)
        elif test == 'lsdd':
            cd = LSDDDrift(x_ref = self.__subsample(self.ref))
            report_dict = cd.predict(self.__subsample(self.cur), return_p_val=True)
            score = report_dict['data']['p_val']
            drifted = report_dict['data']['is_drift']
            my_dict = {
                'drift_score': score,
                'is_drifted': drifted
            }
            if self.showReport:
                self.__saveGlobalReport(building_id, test, score, drifted)

        return my_dict

    # per-column reference vs. current distribution overlay, annotated with p-value/drift status
    def __saveDistributionReport(self, building_id, test, p_vals, is_drift):
        n = len(self.column_names)
        ncols = min(n, 4)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for i, col in enumerate(self.column_names):
            ax = axes[i // ncols][i % ncols]
            ax.hist(self.ref[:, i], bins=30, alpha=0.5, label='reference', density=True)
            ax.hist(self.cur[:, i], bins=30, alpha=0.5, label='current', density=True)
            drifted = bool(is_drift[i])
            ax.set_title(f"{col}\np={p_vals[i]:.3g}  drift={drifted}", color='red' if drifted else 'green', fontsize=9)
            ax.legend(fontsize=7)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis('off')
        fig.suptitle(f"AlibiDetect {test} - building {building_id}")
        fig.tight_layout()
        fig.savefig(self._report_path('alibidetect', f"alibidetect_report_{building_id}_{test}.svg"))
        plt.close(fig)

    # spotdiff/mmd/lsdd yield one global score rather than a per-column one, so summarize it as a single bar
    def __saveGlobalReport(self, building_id, test, p_val, is_drift):
        drifted = bool(is_drift)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(['p-value'], [p_val], color='red' if drifted else 'green')
        ax.axhline(0.05, color='black', linestyle='--', label='threshold (0.05)')
        ax.set_ylim(0, max(1.0, float(p_val) * 1.1))
        ax.set_title(f"AlibiDetect {test} - building {building_id}\ndrift={drifted}", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self._report_path('alibidetect', f"alibidetect_report_{building_id}_{test}.svg"))
        plt.close(fig)

