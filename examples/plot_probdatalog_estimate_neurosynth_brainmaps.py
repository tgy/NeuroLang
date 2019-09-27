# coding: utf-8
r'''
ProbDatalog Neurosynth per-term forward inference brain maps reconstruction
===========================================================================

This example reconstructs the forward inference brain maps associated with each
term in the Neurosynth [1]_ database.

.. [1] Yarkoni et al., "Large-scale automated synthesis of human functional
       neuroimaging data"

'''

import os
from collections import defaultdict
import random

import neurosynth as ns
from neurosynth import Dataset
from nilearn import plotting
import numpy as np

import neurolang as nl
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.expression_walker import ExpressionBasicEvaluator
from neurolang.datalog.expressions import Implication, Fact, Conjunction
from neurolang.datalog.instance import SetInstance
from neurolang.probabilistic.probdatalog import (
    ProbDatalogProgram, ProbFact, full_observability_parameter_estimation
)

random.seed(42)

if not os.path.isfile('database.txt'):
    ns.dataset.download(path='.', unpack=True)
dataset = Dataset('database.txt')
dataset.add_features('features.txt')
image_data = dataset.get_image_data()

study_ids = set(dataset.feature_table.data.index)
n_voxels = image_data.shape[0]
n_studies = image_data.shape[1]
terms_with_decent_study_count = set(
    dataset.feature_table.get_features_by_ids(
        dataset.feature_table.data.index, threshold=0.01
    )
)
n_terms = len(terms_with_decent_study_count)

selected_voxel_ids = set(random.choices(range(n_voxels), k=2))
selected_terms = set(random.choices(list(terms_with_decent_study_count), k=1))
selected_study_ids = set(random.choices(list(study_ids), k=2))


def study_id_to_idx(study_id):
    return np.argwhere(dataset.feature_table.data.index == study_id)[0][0]


def get_study_terms(study_id):
    mask = dataset.feature_table.data.ix[study_id] > 0.01
    return set(dataset.feature_table.data.columns[mask]) & selected_terms


def get_study_reported_voxel_ids(study_id):
    study_idx = study_id_to_idx(study_id)
    return (
        set(np.argwhere(image_data[:, study_idx] > 0).flatten()) &
        selected_voxel_ids
    )


def build_interpretation(study_id):
    terms = get_study_terms(study_id)
    voxel_ids = get_study_reported_voxel_ids(study_id)
    return SetInstance({
        Activation:
        frozenset(
            set.union(
                *([set()] + [{(Constant[int](voxel_id), Constant[str](term))
                              for voxel_id in voxel_ids}
                             for term in terms])
            )
        )
    })


class ProbDatalog(ProbDatalogProgram, ExpressionBasicEvaluator):
    pass


Activation = Symbol('Activation')
DoesActivate = Symbol('DoesActivate')
Voxel = Symbol('Voxel')
Term = Symbol('Term')
v = Symbol('v')
t = Symbol('t')

program = ProbDatalog()
for term in selected_terms:
    program.walk(Fact(Term(Constant[str](term))))
for voxel_id in selected_voxel_ids:
    program.walk(Fact(Voxel(Constant[int](voxel_id))))
program.walk(
    Implication(
        DoesActivate(v, t), Conjunction([Voxel(v),
                                         Term(t),
                                         Activation(v, t)])
    )
)
for term in selected_terms:
    for voxel_id in selected_voxel_ids:
        parameter = Symbol(f'p_{term}_{voxel_id}')
        atom = Activation(Constant[int](voxel_id), Constant[str](term))
        probfact = ProbFact(parameter, atom)
        program.walk(probfact)

interpretations = [
    build_interpretation(study_id) for study_id in selected_study_ids
]

estimations = full_observability_parameter_estimation(program, interpretations)