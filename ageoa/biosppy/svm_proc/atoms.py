from __future__ import annotations
"""Auto-generated atom wrappers following the ageoa pattern."""


from typing import Any
import numpy as np
import torch
import jax
import jax.numpy as jnp
import haiku as hk

import networkx as nx  # type: ignore
import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_assess_classification, witness_assess_runs, witness_combination, witness_cross_validation, witness_get_auth_rates, witness_get_id_rates, witness_get_subject_results, witness_majority_rule

# Witness functions should be imported from the generated witnesses module

@register_atom(witness_get_auth_rates)
@icontract.require(lambda TP: TP is not None, "TP cannot be None")
@icontract.require(lambda FP: FP is not None, "FP cannot be None")
@icontract.require(lambda TN: TN is not None, "TN cannot be None")
@icontract.require(lambda FN: FN is not None, "FN cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Get Auth Rates output must not be None")
def get_auth_rates(TP: Any, FP: Any, TN: Any, FN: Any, thresholds: Any) -> Any:
    """Compute authentication rates from the confusion matrix.

Parameters
----------
TP : array
    True Positive counts for each classifier threshold.
FP : array
    False Positive counts for each classifier threshold.
TN : array
    True Negative counts for each classifier threshold.
FN : array
    False Negative counts for each classifier threshold.
thresholds : array
    Classifier thresholds.

Returns
-------
Acc : array
    Accuracy at each classifier threshold.
TAR : array
    True Accept Rate at each classifier threshold.
FAR : array
    False Accept Rate at each classifier threshold.
FRR : array
    False Reject Rate at each classifier threshold.
TRR : array
    True Reject Rate at each classifier threshold.
EER : array
    Equal Error Rate points, with format (threshold, rate).
Err : array
    Error rate at each classifier threshold.
PPV : array
    Positive Predictive Value at each classifier threshold.
FDR : array
    False Discovery Rate at each classifier threshold.
NPV : array
    Negative Predictive Value at each classifier threshold.
FOR : array
    False Omission Rate at each classifier threshold.
MCC : array
    Matthrews Correlation Coefficient at each classifier threshold.

    Args:
        TP: Input data.
        FP: Input data.
        TN: Input data.
        FN: Input data.
        thresholds: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_get_id_rates)
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.require(lambda M: M is not None, "M cannot be None")
@icontract.require(lambda R: R is not None, "R cannot be None")
@icontract.require(lambda N: N is not None, "N cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Get Id Rates output must not be None")
def get_id_rates(H: Any, M: Any, R: Any, N: Any, thresholds: Any) -> Any:
    """Compute identification rates from the confusion matrix.

Parameters
----------
H : array
    Hit counts for each classifier threshold.
M : array
    Miss counts for each classifier threshold.
R : array
    Reject counts for each classifier threshold.
N : int
    Number of test samples.
thresholds : array
    Classifier thresholds.

Returns
-------
Acc : array
    Accuracy at each classifier threshold.
Err : array
    Error rate at each classifier threshold.
MR : array
    Miss Rate at each classifier threshold.
RR : array
    Reject Rate at each classifier threshold.
EID : array
    Error of Identification points, with format (threshold, rate).
EER : array
    Equal Error Rate points, with format (threshold, rate).

    Args:
        H: Input data.
        M: Input data.
        R: Input data.
        N: Input data.
        thresholds: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_get_subject_results)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda subject: subject is not None, "subject cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.require(lambda subjects: subjects is not None, "subjects cannot be None")
@icontract.require(lambda subject_dict: subject_dict is not None, "subject_dict cannot be None")
@icontract.require(lambda subject_idx: subject_idx is not None, "subject_idx cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Get Subject Results output must not be None")
def get_subject_results(results: Any, subject: Any, thresholds: Any, subjects: Any, subject_dict: Any, subject_idx: Any) -> Any:
    """Compute authentication and identification performance metrics for a
given subject.

Parameters
----------
results : dict
    Classification results.
subject : hashable
    True subject label.
thresholds : array
    Classifier thresholds.
subjects : list
    Target subject classes.
subject_dict : bidict
    Subject-label conversion dictionary.
subject_idx : list
    Subject index.

Returns
-------
assessment : dict
    Authentication and identification results.

    Args:
        results: Input data.
        subject: Input data.
        thresholds: Input data.
        subjects: Input data.
        subject_dict: Input data.
        subject_idx: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_assess_classification)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Assess Classification output must not be None")
def assess_classification(results: Any, thresholds: Any) -> Any:
    """Assess the performance of a biometric classification test.

Parameters
----------
results : dict
    Classification results.
thresholds : array
    Classifier thresholds.

Returns
-------
assessment : dict
    Classification assessment.

    Args:
        results: Input data.
        thresholds: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_assess_runs)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda subjects: subjects is not None, "subjects cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Assess Runs output must not be None")
def assess_runs(results: Any, subjects: Any) -> Any:
    """Assess the performance of multiple biometric classification runs.

Parameters
----------
results : list
    Classification assessment for each run.
subjects : list
    Common target subject classes.

Returns
-------
assessment : dict
    Global classification assessment.

    Args:
        results: Input data.
        subjects: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_combination)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda weights: weights is not None, "weights cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Combination output must not be None")
def combination(results: Any, weights: Any) -> Any:
    """Combine results from multiple classifiers.

Parameters
----------
results : dict
    Results for each classifier.
weights : dict, optional
    Weight for each classifier.

Returns
-------
decision : object
    Consensus decision.
confidence : float
    Confidence estimate of the decision.
counts : array
    Weight for each possible decision outcome.
classes : array
    List of possible decision outcomes.

    Args:
        results: Input data.
        weights: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_majority_rule)
@icontract.require(lambda labels: labels is not None, "labels cannot be None")
@icontract.require(lambda random: random is not None, "random cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Majority Rule output must not be None")
def majority_rule(labels: Any, random: Any) -> Any:
    """Determine the most frequent class label.

Parameters
----------
labels : array, list
    List of clas labels.
random : bool, optional
    If True, will choose randomly in case of tied classes, otherwise the
    first element is chosen.

Returns
-------
    decision : object
        Consensus decision.
    count : int
        Number of elements of the consensus decision.

    Args:
        labels: Input data.
        random: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")

@register_atom(witness_cross_validation)
@icontract.require(lambda labels: labels is not None, "labels cannot be None")
@icontract.require(lambda n_iter: n_iter is not None, "n_iter cannot be None")
@icontract.require(lambda test_size: test_size is not None, "test_size cannot be None")
@icontract.require(lambda train_size: train_size is not None, "train_size cannot be None")
@icontract.require(lambda random_state: random_state is not None, "random_state cannot be None")
@icontract.ensure(lambda result, **kwargs: result is not None, "Cross Validation output must not be None")
def cross_validation(labels: Any, n_iter: Any, test_size: Any, train_size: Any, random_state: Any) -> Any:
    """Return a Cross Validation (CV) iterator.

Wraps the StratifiedShuffleSplit iterator from sklearn.model_selection.
This iterator returns stratified randomized folds, which preserve the
percentage of samples for each class.

Parameters
----------
labels : list, array
    List of class labels for each data sample.
n_iter : int, optional
    Number of splitting iterations.
test_size : float, int, optional
    If float, represents the proportion of the dataset to include in the
    test split; if int, represents the absolute number of test samples.
train_size : float, int, optional
    If float, represents the proportion of the dataset to include in the
    train split; if int, represents the absolute number of train samples.
random_state : int, RandomState, optional
    The seed of the pseudo random number generator to use when shuffling
    the data.

Returns
-------
cv : CV iterator
    Cross Validation iterator.

    Args:
        labels: Input data.
        n_iter: Input data.
        test_size: Input data.
        train_size: Input data.
        random_state: Input data.

    Returns:
        Result data.
    """
    raise NotImplementedError("Wire to original implementation")
