import sys

sys.path.append("../mutation_effect_analysis")

import random
import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics


def get_auc_roc_score(non_nan_result_df: pd.DataFrame, verbose=False):
    larger_means_positive_class = True
    df = non_nan_result_df.copy(deep=True)
    auc_roc_score = sklearn_metrics.roc_auc_score(df["class_numeric"], df["pred"])
    if auc_roc_score < 0.5:
        larger_means_positive_class = False
    if verbose:
        print(f"\tAUC-ROC: {auc_roc_score:.3f}")
    return auc_roc_score, larger_means_positive_class


def get_auc_pr_score(non_nan_result_df: pd.DataFrame, verbose=False):
    df = non_nan_result_df.copy(deep=True)
    precision, recall, thresholds = sklearn_metrics.precision_recall_curve(
        df["class_numeric"], df["pred"], pos_label=1
    )
    auc_pr_score = sklearn_metrics.auc(recall, precision)

    if verbose:
        print(f"\tAUC-PR: {auc_pr_score:.3f}")
    return auc_pr_score, precision, recall, thresholds


def get_f1max_and_th(precisions, recalls, thresholds, verbose=False):
    zero_indices = [
        i
        for i in range(precisions.shape[0])
        if precisions[i] == 0.0 and recalls[i] == 0.0
    ]

    # if precision and recall both are 0, f1=nan
    # so removing those entries where both precision and recall are 0.
    precisions = np.delete(precisions, zero_indices)
    recalls = np.delete(recalls, zero_indices)
    thresholds = np.delete(thresholds, zero_indices)

    f1_scores = (2 * recalls * precisions) / (recalls + precisions)
    th_max = thresholds[np.argmax(f1_scores)]
    f1_max = np.max(f1_scores)
    if verbose:
        print(f"\tBest F1-Score: {f1_max:.3f} at threshold: {th_max:.3f}")
    return f1_max, th_max, precisions, recalls, thresholds


def get_f1max_and_th_(non_nan_result_df: pd.DataFrame, verbose=False):
    df = non_nan_result_df.copy(deep=True)
    f1_scores, precisions, recalls = [], [], []
    thresholds = np.arange(0, 1, 0.01)

    for th in thresholds:
        df.loc[df["pred"] >= th, "class_numeric_pred"] = 1
        df.loc[df["pred"] < th, "class_numeric_pred"] = 0
        prec = sklearn_metrics.precision_score(
            df["class_numeric"],
            df["class_numeric_pred"],
            pos_label=1,
            average="weighted",
        )
        rec = sklearn_metrics.recall_score(
            df["class_numeric"],
            df["class_numeric_pred"],
            pos_label=1,
            average="weighted",
        )
        f1 = sklearn_metrics.f1_score(
            df["class_numeric"],
            df["class_numeric_pred"],
            pos_label=1,
            average="weighted",
        )

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    f1_max = np.max(f1_scores)
    th_max = thresholds[np.argmax(f1_scores)]
    if verbose:
        print(f"\tcBest F1-Score: {f1_max:.3f} at threshold: {th_max:.3f}")
    return f1_max, th_max, precisions, recalls, thresholds


def get_precision_score(non_nan_result_df: pd.DataFrame, th, verbose=False):
    df = non_nan_result_df.copy(deep=True)
    df.loc[df["pred"] >= th, "class_numeric_pred"] = 1
    df.loc[df["pred"] < th, "class_numeric_pred"] = 0
    precision_score = sklearn_metrics.precision_score(
        df["class_numeric"], df["class_numeric_pred"], zero_division=0
    )

    if verbose:
        print(f"\tPrecision score: {precision_score:.3f} at threshold: {th:.3f}")
    return precision_score


def get_recall_score(non_nan_result_df: pd.DataFrame, th, verbose=False):
    df = non_nan_result_df.copy(deep=True)
    df.loc[df["pred"] >= th, "class_numeric_pred"] = 1
    df.loc[df["pred"] < th, "class_numeric_pred"] = 0
    recall_score = sklearn_metrics.recall_score(
        df["class_numeric"], df["class_numeric_pred"]
    )

    if verbose:
        print(f"\tRecall score: {recall_score:.3f} at threshold: {th:.3f}")
    return recall_score


def get_accuracy_score(non_nan_result_df: pd.DataFrame, th, verbose=False):
    df = non_nan_result_df.copy(deep=True)
    df.loc[df["pred"] >= th, "class_numeric_pred"] = 1
    df.loc[df["pred"] < th, "class_numeric_pred"] = 0
    accuracy_score = sklearn_metrics.accuracy_score(
        df["class_numeric"], df["class_numeric_pred"]
    )

    if verbose:
        print(f"\tAccuracy score: {accuracy_score:.3f} at threshold: {th:.3f}")
    return accuracy_score


def get_matthews_corrcoef(non_nan_result_df: pd.DataFrame, th, verbose=False):
    df = non_nan_result_df.copy(deep=True)
    df.loc[df["pred"] >= th, "class_numeric_pred"] = 1
    df.loc[df["pred"] < th, "class_numeric_pred"] = 0
    mcc_score = sklearn_metrics.matthews_corrcoef(
        df["class_numeric"], df["class_numeric_pred"]
    )

    if verbose:
        print(f"\tMCC score: {mcc_score:.3f} at threshold: {th:.3f}")
    return mcc_score


def get_balanced_accuracy_score(non_nan_result_df: pd.DataFrame, th, verbose=False):
    df = non_nan_result_df.copy(deep=True)
    df.loc[df["pred"] >= th, "class_numeric_pred"] = 1
    df.loc[df["pred"] < th, "class_numeric_pred"] = 0
    balanced_accuracy_score = sklearn_metrics.balanced_accuracy_score(
        df["class_numeric"], df["class_numeric_pred"]
    )

    if verbose:
        print(
            f"\tBalanced accuracy score: {balanced_accuracy_score:.3f} at threshold: {th:.3f}"
        )
    return balanced_accuracy_score


from scipy.stats import ks_2samp


def get_KS_test_score(df: pd.DataFrame, verbose=False):
    pos_cls_preds = df[df["class_numeric"] == 1]["pred"]
    neg_cls_preds = df[df["class_numeric"] == 0]["pred"]

    res = ks_2samp(pos_cls_preds, neg_cls_preds)
    if verbose:
        print(
            f"\tKS-test score. statistic: {res.statistic:.3f}, p-value: {res.pvalue:.3f}"
        )
    return res.statistic, res.pvalue


# ------------------------ performance computation helper functions ------------------------------
# by default, we consider larger means positive class (here Effect/Knock-out), but the following models have specific opposite meaning

metrics = [
    "AUC-ROC",
    "AUC-PR",
    "F1-max",
    "Th-max",
    "Precision",
    "Recall",
    "Accuracy",
    "Balanced-accuracy",
    "MCC",
]


def sample_positive_and_negative_data_points(
    df, method_name, positive_cls, negative_cls, n_samples=None
):
    df = df[(df["class"] == positive_cls) | (df["class"] == negative_cls)].copy()
    df = df[~pd.isna(df[method_name])]  # taking only non-NAN values

    positive_cls_df = df[df["class"] == positive_cls].copy()
    negative_cls_df = df[df["class"] == negative_cls].copy()

    positive_cls_df["class_numeric"] = 1
    negative_cls_df["class_numeric"] = 0

    if n_samples is not None:
        n_samples_pos_cls = min(n_samples, positive_cls_df.shape[0])
        n_samples_neg_cls = min(n_samples, negative_cls_df.shape[0])
        positive_cls_df = positive_cls_df.sample(n=n_samples_pos_cls).copy()
        negative_cls_df = negative_cls_df.sample(n=n_samples_neg_cls).copy()

    else:
        # setting #-samples as the minority classes
        n_samples = min(positive_cls_df.shape[0], negative_cls_df.shape[0])
        positive_cls_df = positive_cls_df.sample(n=n_samples).copy()
        negative_cls_df = negative_cls_df.sample(n=n_samples).copy()

    sampled_df = pd.concat([positive_cls_df, negative_cls_df])
    return sampled_df


# from utils.data_dicts import methods_smaller_means_damaging_from_paper
# def calibrate_prediction_scores_direction(df, method_name):
#     if method_name in methods_smaller_means_damaging_from_paper:
#         df['pred'] = df['pred'].multiply(-1)

#     auc_roc_score, larger_means_positive_class = get_auc_roc_score(df)
#     if not larger_means_positive_class:
#         df['pred'] = df['pred'].multiply(-1)
#     return df, auc_roc_score


def get_pathogenic_analysis_threshold(method_name, home_dir=""):
    patho_performance_metrics_df = pd.read_csv(
        home_dir
        + f"data/performance_analysis_minority_cls/patho_Pathogenic_vs_Neutral.tsv",
        sep="\t",
    )  # performance_analysis, performance_analysis_minority_cls, performance_analysis_alphamissense, performance_analysis_alphamissense_minority_cls
    patho_th_max = patho_performance_metrics_df[
        patho_performance_metrics_df["Models\\Metrics"] == method_name
    ]["Th-max"].values[1]
    patho_th_max = patho_th_max.split("(")[0]
    patho_th_max = float(patho_th_max)
    # print(f"\tComputed th from pathogenic-analysis: {patho_th_max}")
    return patho_th_max


# print(get_pathogenic_analysis_threshold("phastCons17way_primate"))


def compute_performance_metrics(
    df,
    method_name,
    positive_cls,
    negative_cls,
    n_runs=10,
    n_samples=None,
    home_dir="",
    fill_missing_with_median=True,
):
    print(method_name)

    if fill_missing_with_median and method_name != "random_classifier":
        median = df[method_name].median()
        df.loc[
            pd.isna(df[method_name]), method_name
        ] = median  # filling with median in the missing prediction score location

    metric_scores = []
    for i in range(n_runs):
        if method_name == "random_classifier":
            df[method_name] = [random.uniform(0, 1) for i in range(df.shape[0])]

        sampled_df = sample_positive_and_negative_data_points(
            df, method_name, positive_cls, negative_cls, n_samples
        )
        sampled_df["pred"] = sampled_df[method_name].copy()

        # scaling between [0,1] has been done while making unidirectional
        # sampled_df["pred"]=(sampled_df[method_name]-sampled_df[method_name].min())/(sampled_df[method_name].max()-sampled_df[method_name].min()) # scaling prediction scores between [0, 1]

        if method_name in ["phyloP17way_primate", "phastCons17way_primate"]:
            th_max = 0.5
            sampled_df.loc[sampled_df["pred"] >= th_max, "pred"] = 1
            sampled_df.loc[sampled_df["pred"] < th_max, "pred"] = 0

        # unidirectional process already did the calibration
        # sampled_df, auc_roc_score =  calibrate_prediction_scores_direction(sampled_df, method_name)

        auc_roc_score, _ = get_auc_roc_score(sampled_df)
        # ks_statistic, ks_pvalue = get_KS_test_score(sampled_df)
        auc_pr_score, precisions, recalls, thresholds = get_auc_pr_score(sampled_df)
        f1_max, th_max, precisions, recalls, thresholds = get_f1max_and_th(
            precisions, recalls, thresholds
        )

        if positive_cls == "Likely-pathogenic":
            th_max = get_pathogenic_analysis_threshold(method_name, home_dir)
        if method_name in ["phyloP17way_primate", "phastCons17way_primate"]:
            th_max = 0.5

        precision = get_precision_score(sampled_df, th_max)
        recall = get_recall_score(sampled_df, th_max)
        accuracy = get_accuracy_score(sampled_df, th_max)
        balanced_accuracy = get_balanced_accuracy_score(sampled_df, th_max)
        mcc = get_matthews_corrcoef(sampled_df, th_max)

        metric_scores.append(
            [
                auc_roc_score,
                auc_pr_score,
                f1_max,
                th_max,
                precision,
                recall,
                accuracy,
                balanced_accuracy,
                mcc,
            ]
        )
        # print()
        # if i==0: break
    return metric_scores


def write_metrics_outputs(performance_scores_dict, output_file):
    out = open(output_file, "w")
    out.write("Models\\Metrics")
    for metric in metrics:
        out.write(f"\t{metric}")
    out.write("\n")

    for model_name, performance_scores in performance_scores_dict.items():
        out.write(f"{model_name}")
        for scores in performance_scores:
            for score in scores:
                out.write(f"\t{score:.3f}")
            out.write("\n")
        out.write("\n")

    for model_name, performance_scores in performance_scores_dict.items():
        out.write(f"{model_name}")
        avg_scores = np.mean(performance_scores, axis=0)
        std_scores = np.std(performance_scores, axis=0)
        for i in range(len(avg_scores)):
            out.write(f"\t{avg_scores[i]:.3f}({std_scores[i]:.3f})")
        out.write("\n")
    out.close()
