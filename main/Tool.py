from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import *
from evidently.legacy.pipeline.column_mapping import ColumnMapping

import nannyml as nml # pip install nannyml

from alibi_detect.cd import KSDrift, CVMDrift, SpotTheDiffDrift, MMDDrift, LSDDDrift

from frouros.detectors.data_drift.batch import (
    KSTest, CVMTest, AndersonDarlingTest, ChiSquareTest, MannWhitneyUTest, WelchTTest,
    BWSTest, KuiperTest, PSI, KL, JS, EMD, HellingerDistance, EnergyDistance,
    BhattacharyyaDistance, HINormalizedComplement, MMD as FrourosMMD,
)
from frouros.callbacks.batch import PermutationTestDistanceBased

from river.drift import ADWIN, KSWIN, PageHinkley

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
    CHI2 = 15 # Chi-Square Test
    BWS = 16 # Baumgartner-Weiss-Schindler Test
    KUIPER = 17 # Kuiper's Test
    BHATTACHARYYA = 18 # Bhattacharyya Distance
    HI = 19 # Histogram Intersection (normalized complement)
    ADWIN_M = 20 # ADWIN change detector (River, repurposed from concept- to covariate-drift monitoring)
    KSWIN_M = 21 # KSWIN windowed K-S detector (River)
    PAGE_HINKLEY = 22 # Page-Hinkley change detector (River)

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
            ref_sample = self.__subsample(self.ref)
            cur_sample = self.__subsample(self.cur)
            cd = MMDDrift(x_ref = ref_sample)
            report_dict = cd.predict(cur_sample, return_p_val=True)
            score = report_dict['data']['p_val']
            drifted = report_dict['data']['is_drift']
            my_dict = {
                'drift_score': score,
                'is_drifted': drifted,
                # MMD builds an O(n^2) kernel matrix, so ref/cur are capped at MAX_KERNEL_SAMPLES;
                # compare these counts to n_ref_total/n_cur_total before treating runtime/CPU/RAM
                # numbers for this test as comparable to tools that ran on the full window
                'n_ref_used': len(ref_sample),
                'n_cur_used': len(cur_sample),
                'n_ref_total': len(self.ref),
                'n_cur_total': len(self.cur),
            }
            if self.showReport:
                self.__saveGlobalReport(building_id, test, score, drifted)
        elif test == 'lsdd':
            ref_sample = self.__subsample(self.ref)
            cur_sample = self.__subsample(self.cur)
            cd = LSDDDrift(x_ref = ref_sample)
            report_dict = cd.predict(cur_sample, return_p_val=True)
            score = report_dict['data']['p_val']
            drifted = report_dict['data']['is_drift']
            my_dict = {
                'drift_score': score,
                'is_drifted': drifted,
                'n_ref_used': len(ref_sample),
                'n_cur_used': len(cur_sample),
                'n_ref_total': len(self.ref),
                'n_cur_total': len(self.cur),
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

class Frouros(Tool):
    ALPHA = 0.05
    PERMUTATIONS = 100          # permutation count for the univariate distance-based tests (cheap, O(n) per permutation)
    # MMD builds an O(n^2) kernel matrix like AlibiDetect's, but frouros computes it in chunks
    # (chunk_size) instead of materializing the full matrix, so ref/cur are NOT subsampled here --
    # unlike AlibiDetect.MMD/LSDD, this test runs on the full window. The tradeoff is fewer
    # permutations (each one re-chunks the O(n^2) kernel, which is expensive at full scale).
    MMD_CHUNK_SIZE = 500
    MMD_PERMUTATIONS = 30

    def __init__(self, name, showReport=False):
        super().__init__(name)
        self.showReport = showReport
        self.methods = {METHODS.KOLMOGOROV_SMIRNOV, METHODS.CVM, METHODS.AD, METHODS.MWURT, METHODS.TT,
                         METHODS.CHI2, METHODS.BWS, METHODS.KUIPER,
                         METHODS.PSI, METHODS.KLD, METHODS.JSD, METHODS.HD, METHODS.ED, METHODS.BHATTACHARYYA,
                         METHODS.HI, METHODS.MMD}

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
                my_dict['K-S Test'] = self.__runStatTest(building_id, 'kolmogorov_smirnov', KSTest)
            elif test == METHODS.CVM:
                my_dict['Cramer-von-Mises'] = self.__runStatTest(building_id, 'cramer_von_mises', CVMTest)
            elif test == METHODS.AD:
                my_dict['Anderson-Darling'] = self.__runStatTest(building_id, 'anderson', AndersonDarlingTest)
            elif test == METHODS.MWURT:
                my_dict['Mann-Whitney U-Rank Test'] = self.__runStatTest(building_id, 'mannw', MannWhitneyUTest)
            elif test == METHODS.TT:
                my_dict['T-Test'] = self.__runStatTest(building_id, 't_test', WelchTTest)
            elif test == METHODS.CHI2:
                my_dict['Chi-Square Test'] = self.__runStatTest(building_id, 'chi_square', ChiSquareTest)
            elif test == METHODS.BWS:
                my_dict['Baumgartner-Weiss-Schindler Test'] = self.__runStatTest(building_id, 'bws', BWSTest)
            elif test == METHODS.KUIPER:
                my_dict["Kuiper's Test"] = self.__runStatTest(building_id, 'kuiper', KuiperTest)
            elif test == METHODS.PSI:
                my_dict['PSI'] = self.__runDistanceTest(building_id, 'psi', PSI)
            elif test == METHODS.KLD:
                my_dict['K-L Divergence'] = self.__runDistanceTest(building_id, 'kl_div', KL)
            elif test == METHODS.JSD:
                my_dict['J-S Distance'] = self.__runDistanceTest(building_id, 'jensenshannon', JS)
            elif test == METHODS.HD:
                my_dict['Hellinger-Distance'] = self.__runDistanceTest(building_id, 'hellinger', HellingerDistance)
            elif test == METHODS.ED:
                my_dict['Energy-Distance'] = self.__runDistanceTest(building_id, 'ed', EnergyDistance)
            elif test == METHODS.BHATTACHARYYA:
                my_dict['Bhattacharyya-Distance'] = self.__runDistanceTest(building_id, 'bhattacharyya', BhattacharyyaDistance)
            elif test == METHODS.HI:
                my_dict['Histogram-Intersection'] = self.__runDistanceTest(building_id, 'hi', HINormalizedComplement, num_bins=20)
            elif test == METHODS.MMD:
                my_dict['Maximum Mean Discrepancy'] = self.__runMMD(building_id)

        return my_dict

    # KSTest/CVMTest/AndersonDarlingTest/... natively return a p-value from compare(), no permutation needed
    def __runStatTest(self, building_id, test, detector_cls):
        p_vals, drifted = [], []
        for i in range(len(self.column_names)):
            det = detector_cls()
            det.fit(X=self.ref[:, i])
            result, _ = det.compare(X=self.cur[:, i])
            p_vals.append(float(result.p_value))
            drifted.append(bool(result.p_value < self.ALPHA))

        my_dict = {}
        for i, col in enumerate(self.column_names):
            my_dict[f"{col}_drift_score"] = p_vals[i]
            my_dict[f"{col}_is_drifted"] = drifted[i]
        if self.showReport:
            self.__saveDistributionReport(building_id, test, p_vals, drifted)
        return my_dict

    # PSI/KL/JS/EMD/... only return a raw distance; attach a permutation-test callback to get a
    # p-value so drift decisions use the same alpha=0.05 rule as the native statistical tests
    def __runDistanceTest(self, building_id, test, detector_cls, **extra_kwargs):
        p_vals, drifted = [], []
        for i in range(len(self.column_names)):
            callback = PermutationTestDistanceBased(num_permutations=self.PERMUTATIONS, num_jobs=-1,
                                                      random_state=0, name='perm')
            det = detector_cls(callbacks=[callback], **extra_kwargs)
            det.fit(X=self.ref[:, i])
            _, logs = det.compare(X=self.cur[:, i])
            p_val = float(logs['perm']['p_value'])
            p_vals.append(p_val)
            drifted.append(bool(p_val < self.ALPHA))

        my_dict = {}
        for i, col in enumerate(self.column_names):
            my_dict[f"{col}_drift_score"] = p_vals[i]
            my_dict[f"{col}_is_drifted"] = drifted[i]
        if self.showReport:
            self.__saveDistributionReport(building_id, test, p_vals, drifted)
        return my_dict

    def __runMMD(self, building_id):
        callback = PermutationTestDistanceBased(num_permutations=self.MMD_PERMUTATIONS, num_jobs=-1,
                                                  random_state=0, name='perm')
        det = FrourosMMD(chunk_size=self.MMD_CHUNK_SIZE, callbacks=[callback])
        det.fit(X=self.ref)
        result, logs = det.compare(X=self.cur)
        p_val = float(logs['perm']['p_value'])
        drifted = bool(p_val < self.ALPHA)
        my_dict = {
            'drift_score': p_val,
            'is_drifted': drifted,
            'distance': float(result.distance),
            # unlike AlibiDetect.MMD, these equal n_ref_total/n_cur_total -- full window, no subsampling
            'n_ref_used': len(self.ref),
            'n_cur_used': len(self.cur),
            'n_ref_total': len(self.ref),
            'n_cur_total': len(self.cur),
        }
        if self.showReport:
            self.__saveGlobalReport(building_id, 'mmd', p_val, drifted)
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
        fig.suptitle(f"Frouros {test} - building {building_id}")
        fig.tight_layout()
        fig.savefig(self._report_path('frouros', f"frouros_report_{building_id}_{test}.svg"))
        plt.close(fig)

    # MMD yields one global score rather than a per-column one, so summarize it as a single bar
    def __saveGlobalReport(self, building_id, test, p_val, is_drift):
        drifted = bool(is_drift)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(['p-value'], [p_val], color='red' if drifted else 'green')
        ax.axhline(0.05, color='black', linestyle='--', label='threshold (0.05)')
        ax.set_ylim(0, max(1.0, float(p_val) * 1.1))
        ax.set_title(f"Frouros {test} - building {building_id}\ndrift={drifted}", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self._report_path('frouros', f"frouros_report_{building_id}_{test}.svg"))
        plt.close(fig)

class River(Tool):
    # River's detectors are online change detectors, not two-sample batch tests: each is warmed
    # up by replaying `ref` through .update(), then replays `cur` one value at a time, flagging
    # `drift_detected` the moment the stream diverges from what it has seen so far. There is no
    # p-value. This is a repurposing of River's streaming API into D3Bench's ref-vs-cur batch
    # pattern -- River is designed to consume one continuous production stream, not two static
    # windows, and (unlike the other tools here) these detectors are usually run on a single
    # signal (e.g. model error) rather than per feature. Applying them per-column below is the
    # closest analogue to what Evidently/AlibiDetect/Frouros report per column.
    #
    # Reported per column instead of a p-value:
    #   drift_score = fraction of `cur` samples that triggered a drift signal (higher = more drift,
    #                 opposite direction from the p-values reported by every other tool here)
    #   first_drift_index = position in `cur` where drift was first flagged (None if never)
    def __init__(self, name, showReport=False):
        super().__init__(name)
        self.showReport = showReport
        self.methods = {METHODS.KSWIN_M, METHODS.ADWIN_M, METHODS.PAGE_HINKLEY}

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
            if test == METHODS.KSWIN_M:
                my_dict['KSWIN'] = self.__runStreamingTest(building_id, 'kswin', KSWIN)
            elif test == METHODS.ADWIN_M:
                my_dict['ADWIN'] = self.__runStreamingTest(building_id, 'adwin', ADWIN)
            elif test == METHODS.PAGE_HINKLEY:
                my_dict['Page-Hinkley'] = self.__runStreamingTest(building_id, 'page_hinkley', PageHinkley)

        return my_dict

    def __runStreamingTest(self, building_id, test, detector_cls):
        scores, drifted, first_idx = [], [], []
        for i in range(len(self.column_names)):
            det = detector_cls()
            for x in self.ref[:, i]:
                det.update(float(x))

            n_flags = 0
            idx = None
            for j, x in enumerate(self.cur[:, i]):
                det.update(float(x))
                if det.drift_detected:
                    n_flags += 1
                    if idx is None:
                        idx = j
            scores.append(n_flags / len(self.cur))
            drifted.append(n_flags > 0)
            first_idx.append(idx)

        my_dict = {}
        for i, col in enumerate(self.column_names):
            my_dict[f"{col}_drift_score"] = scores[i]
            my_dict[f"{col}_is_drifted"] = drifted[i]
            my_dict[f"{col}_first_drift_index"] = first_idx[i]
        if self.showReport:
            self.__saveDistributionReport(building_id, test, scores, drifted)
        return my_dict

    # per-column reference vs. current distribution overlay, annotated with the flagged fraction
    def __saveDistributionReport(self, building_id, test, scores, is_drift):
        n = len(self.column_names)
        ncols = min(n, 4)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for i, col in enumerate(self.column_names):
            ax = axes[i // ncols][i % ncols]
            ax.hist(self.ref[:, i], bins=30, alpha=0.5, label='reference', density=True)
            ax.hist(self.cur[:, i], bins=30, alpha=0.5, label='current', density=True)
            drifted = bool(is_drift[i])
            ax.set_title(f"{col}\nflagged={scores[i]:.3g}  drift={drifted}", color='red' if drifted else 'green', fontsize=9)
            ax.legend(fontsize=7)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis('off')
        fig.suptitle(f"River {test} - building {building_id}")
        fig.tight_layout()
        fig.savefig(self._report_path('river', f"river_report_{building_id}_{test}.svg"))
        plt.close(fig)

