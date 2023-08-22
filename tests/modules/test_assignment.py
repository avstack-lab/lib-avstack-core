# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-01-11
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-21
# @Description:
"""

"""
from functools import partial

import numpy as np
from avstack.modules import assignment
from avstack import datastructs as ds


# Set up cost matrix -- see Blackman P. 347
A_blackman = np.array([[    10,      5,      8,     9],
                      [     7, np.inf,     20, np.inf],
                      [np.inf,     21, np.inf, np.inf],
                      [np.inf,     15,     17, np.inf],
                      [np.inf, np.inf,     16,     22]])


def test_JVC_linear_sum():
    A = np.array([[np.inf, -10, -1],
                  [np.inf, -1, -30],
                  [np.inf, 10, 10]])
    assign_sol = assignment.gnn_single_frame_assign(A, algorithm='JVC')
    assert assign_sol.cost == -40
    assert assign_sol.same(ds.OneEdgeBipartiteGraph({0:1, 1:2}, A.shape[0], A.shape[1], -40))
    assert assign_sol.assignment_tuples == ((0,1), (1,2))

    # test the blackman matrix
    assign_sol = assignment.gnn_single_frame_assign(A_blackman)
    assert assign_sol.cost == 47
    assert assign_sol.assignment_tuples == tuple(sorted(((1,0), (3,1), (4,2), (0,3))))
    assert assign_sol.unassigned_rows == (2,)
    assert assign_sol.unassigned_cols == ()


def test_n_best_assignment():
    # n-best solutions
    assign_sol = assignment.gnn_single_frame_assign(A_blackman)
    assign_n_best = assignment.n_best_solutions(A_blackman, n=5, verbose=False)
    cost_true = [47, 51, 52, 53, 54]
    for i, (c, a) in enumerate(zip(cost_true, assign_n_best)):
        if i == 0:
            assert a == assign_sol
        assert a.cost == c

    # n-best solutions with lone trk
    A_lone_trk = A_blackman.copy()
    A_lone_trk[0,3] = 10
    A_lone_trk[1,0] = np.inf
    A_lone_trk[4,3] = np.inf
    assign_sol = assignment.gnn_single_frame_assign(A_lone_trk)
    assert assign_sol.unassigned_cols == (3,)
    assign_n_best = assignment.n_best_solutions(A_lone_trk, n=4, verbose=False)


def test_assignment_class():
    assignments = [(1,2), (0,1), (3,4)]
    idx_lone_det = [2]
    idx_lone_trk = [0, 3]
    row_to_col = {a[0]:a[1] for a in assignments}
    assign_sol = ds.OneEdgeBipartiteGraph(row_to_col, 4, 5, 10)
    assert assign_sol.assigns_by_col(2) == (1,)
    assert assign_sol.assigns_by_row(3) == (4,)


def test_pda_assignment():
    A = A_blackman
    A = np.concatenate((A, np.inf*np.ones((A.shape[0],1))), axis=1)

    PD = 0.9
    BETA_FT = 0.0001
    BETA_NT = 15*BETA_FT/PD
    BETA = BETA_FT + BETA_NT

    gate_map = {c:[r[0] for r in np.argwhere(A[:,c] < np.inf)] for c in range(A.shape[1])}
    d2_map = {c:{r:A[r,c] for r in gate_map[c]} for c in gate_map}
    S = np.diag(2*np.ones((3,)))
    S_map = {c:S for c in gate_map}
    assign_sol = assignment.pda_single_frame_assign(gate_map, d2_map, S_map, PD, BETA, A.shape[0], A.shape[1])
    assert set(assign_sol.iterate_over('rows').keys()) == set(range(A.shape[0]))
    assert set(assign_sol.iterate_over('cols').keys()) == set(range(A.shape[1])).difference({4})
    for c, rs in assign_sol.iterate_over('cols').items():
        assert 0 < sum([w for w in rs.values()]) <= 1.0


def test_jpda_combinatorics():
    """Example on blackman P. 355"""
    d2_map = {0:{0:1, 1:2, 2:4}, 1:{1:2.5, 2:3}}
    gate_map = {0:[0,1,2], 1:[1,2]}
    S_map = {0:np.diag([1,1]), 1:np.diag([1,1])}
    p_result = [0.011, 0.086, 0.053, 0.019, 0.041, 0.306, 0.068, 0.032, 0.239, 0.145]
    PD = 0.7
    beta = 0.03
    nrows = 3
    ncols = 2
    p_H_partial = partial(assignment.p_H, d2_map, S_map, PD, beta)
    best_sols, probs = assignment._jpda_via_combinatorics(gate_map, p_H_partial, nrows, ncols)
    assert np.isclose(sum(probs), 1.0)
    assert len(best_sols) == 10
    assert len(probs) == 10
    sp = sorted(probs)
    sr = sorted(p_result)
    assert np.allclose(sp, sr, atol=1e-2)


def test_jpda_combinatorics_assignment():
    """Example on blackman P. 355"""
    d2_map = {0:{0:1, 1:2, 2:4}, 1:{1:2.5, 2:3}}
    gate_map = {0:[0,1,2], 1:[1,2]}
    S_map = {0:np.diag([1,1]), 1:np.diag([1,1])}
    PD = 0.7
    beta = 0.03
    nrows = 3
    ncols = 2
    assign_sol = assignment.jpda_single_frame_assign(gate_map, d2_map, S_map,
        PD=PD, beta=beta, nrows=nrows, ncols=ncols,method='combinatorial')
    col_to_row = assign_sol.iterate_over('col')
    assert np.isclose(col_to_row[0][0], 0.631, atol=1e-2)
    assert np.isclose(col_to_row[0][1], 0.198, atol=1e-2)
    assert np.isclose(col_to_row[0][2], 0.087, atol=1e-2)
    assert np.isclose(col_to_row[1][1], 0.415, atol=1e-2)
    assert np.isclose(col_to_row[1][2], 0.416, atol=1e-2)
