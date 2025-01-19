import itertools
import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


## EVALUATION HELPERS FOR LIGHTGBM ##
def evaluate_model_helper(X_test, y_test, lr, exp_info="", viz=False):
    fpr_thresholds = [
        val / 100.0
        for val in [0.4, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    ]
    y_pred_proba = lr.predict_proba(X_test)[:, 1] * 1000
    plt.figure(figsize=(15, 15))
    plt.suptitle(exp_info)

    j = 1
    score_thresholds = []
    recall_values = []
    final_cnf_matrix = None

    for fpr_threshold in fpr_thresholds:
        min_score_threshold = None
        max_recall = None
        max_recall_score_threshold = None
        for i in range(1001):
            y_test_predictions_high_recall = y_pred_proba >= i
            cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)

            TP = cnf_matrix[1, 1]
            FN = cnf_matrix[1, 0]
            FP = cnf_matrix[0, 1]
            TN = cnf_matrix[0, 0]

            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) != 0 else 0

            if fpr <= fpr_threshold:
                min_score_threshold = i
                max_recall = recall
                max_recall_score_threshold = i
                final_cnf_matrix = cnf_matrix
                break

        score_thresholds.append(max_recall_score_threshold)
        recall_values.append(max_recall)

        if viz == True:
            plt.subplot(4, 3, j)
            j += 1

            plot_confusion_matrix(
                final_cnf_matrix,
                classes=["No Fraud", "Fraud"],
                title=f"Threshold = {max_recall_score_threshold}\nFPR<={(fpr_threshold * 100):.2f}% \nRecall={max_recall:.4f}",
            )
    results_fpr = {
        "score_threshold": score_thresholds,
        "fpr %": [f"{fpr * 100.0:.2f}" for fpr in fpr_thresholds],
        "recall %": [f"{(recall * 100.0):.2f}" for recall in recall_values],
    }
    results_fpr_table = pd.DataFrame(results_fpr).sort_values(by="fpr %")
    return results_fpr_table


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def calculate_partial_auc(y_test, probs, fpr_threshold=0.001):
    """
    Calculates the AUC for a specific range of FPR (e.g., FPR <= 0.001).

    Parameters:
        y_test (array-like): True binary labels.
        probs (array-like): Predicted probabilities for the positive class.
        fpr_threshold (float): Maximum value of FPR for which to calculate the AUC.

    Returns:
        float: Partial AUC for FPR <= fpr_threshold.
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)

    # Filter the range for FPR <= fpr_threshold
    mask = fpr <= fpr_threshold
    filtered_fpr = fpr[mask]
    filtered_tpr = tpr[mask]

    # Compute the partial AUC using trapezoidal rule
    partial_auc = np.trapz(filtered_tpr, filtered_fpr)
    return partial_auc


def get_best_params(X_train, y_train):
    def objective(trial):
        # Define the parameter search space
        param = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.7, 0.9),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.7, 0.9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        }

        # Prepare the dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        # Perform cross-validation with early stopping
        cv_results = lgb.cv(
            params=param,
            train_set=train_data,
            nfold=3,
            metrics=["auc"],
            # early_stopping_rounds=50,
            seed=42,
        )

        # Get the best AUC score from cross-validation (mean of validation AUC scores)
        return max(cv_results["valid auc-mean"])

    # Run the optimization process with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Retrieve the best parameters
    best_params = study.best_params

    return best_params


def save_params(best_params, filepath):
    with open(filepath, "w") as file:
        json.dump(best_params, file)


## UTILS ##
def get_fraud_percentage(counter):
    num_fraud = counter[1]
    num_nonfraud = counter[0]
    total = num_fraud + num_nonfraud
    print(
        f"{(num_fraud * 100.0)/total:.4f}% fraud -- {num_fraud} fraud, {num_nonfraud} nonfraud, {total} total"
    )


def duplicate_minority_class(dataset, N):
    fraud_class = dataset[dataset["Class"] == 1]
    nonfraud_class = dataset[dataset["Class"] == 0]

    duplicated_fraud_class = pd.concat([fraud_class] * N, ignore_index=True)
    # Concatenate the duplicated fraud class back to the original dataset
    df_duplicated = pd.concat(
        [nonfraud_class, duplicated_fraud_class], ignore_index=True
    )
    return df_duplicated


# Function to apply SMOTE only to the fraud class
def apply_smote_to_fraud(X, y, target_fraud_ratio):
    # Calculate current number of fraud samples
    fraud_count = y.sum()
    non_fraud_count = len(y) - fraud_count

    # Desired number of fraud samples
    target_fraud_count = int(
        non_fraud_count * target_fraud_ratio / (1 - target_fraud_ratio)
    )

    # Sampling strategy for SMOTE
    smote_strategy = {1: target_fraud_count, 0: non_fraud_count}

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


## OTHER -- MAY GET RID LATER ##
def plot_two_features(all_train, feature_x, feature_y, title_text="", color_col=None):
    """
    Create a scatter plot comparing two features colored by fraud status.

    Args:
        all_train (DataFrame): Dataset containing the features and the "isFraud" column.
        feature_x (str): Name of the first feature.
        feature_y (str): Name of the second feature.
        title_text (str): Title for the plot.

    Returns:
        plotly.graph_objects.Figure: A scatter plot figure.
    """
    fig = px.scatter(
        all_train,
        x=feature_x,
        y=feature_y,
        color=color_col,
        opacity=0.5,
        title=title_text,
    )
    fig.update_layout(
        title_font_color="white",
        legend_title_font_color="yellow",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color="Grey")),
        selector=dict(mode="markers"),
    )
    return fig
