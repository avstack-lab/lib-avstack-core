import itertools
from functools import partial

import numpy as np
from scipy.optimize import linear_sum_assignment

from avstack.datastructs import (
    MultiEdgeBipartiteGraph,
    OneEdgeBipartiteGraph,
    PrioritySet,
    invert_dict_of_list,
)


"""
TODO:
- improve assignment algorithm by eliminating useless or obvious
assignments ahead of calling linear sum algorithm


- clean up assignment after refactor was made to bipartite graph data structure
"""


def build_A_from_distance(objs1, objs2, **kwargs):
    A = np.zeros((len(objs1), len(objs2)))
    for i, o1 in enumerate(objs1):
        for j, o2 in enumerate(objs2):
            try:
                dist = o1.distance(o2, **kwargs)
            except AttributeError:
                dist = np.linalg.norm(o1 - o2)
            A[i, j] = dist
    A += 1e-8
    return A


def build_A_from_iou(boxes1, boxes2, **kwargs):
    A = np.zeros((len(boxes1), len(boxes2)))
    for i, b1 in enumerate(boxes1):
        try:
            b1 = b1.box
        except AttributeError:
            pass
        for j, b2 in enumerate(boxes2):
            try:
                b2 = b2.box
            except AttributeError:
                pass
            iou = b1.IoU(b2, **kwargs)
            A[i, j] = -iou
    return A


def gnn_single_frame_assign(
    A, algorithm="JVC", all_assigned=False, cost_threshold=np.inf
):
    """
    Run linear-sum assignment algorithm

    Rows are considered "detections" and columns are consisdered "tracks"
    (or truths). Thus, any detection not assigned to a track may either be a
    false positive or a new object not yet tracked. Similarly, any track not
    assigned a detection was either a false negative or the object has been
    removed from the frame. For clarity, these are denoted "lone_det" and
    "lone_trk"

    INPUTS
    :A -- cost matrix --> trying to minimize cost
    :algorithm -- the name of the assignment algorithm
    :all_assigned -- boolean for if all entries need an assignment or if we can
        allow the algorithm to not assign for high cost
    """

    assert algorithm in ["JVC"]
    assignments = []
    A = A.copy()
    nrows = A.shape[0] if len(A.shape) > 0 else 0
    ncols = A.shape[1] if len(A.shape) == 2 else 0
    if A.size == 0:
        return OneEdgeBipartiteGraph({}, nrows, ncols, 0)

    # Apply threshold
    A[A >= cost_threshold] = np.inf

    # Eliminate useless ones
    if not all_assigned:
        idx_lone_det = [i for i in range(A.shape[0]) if np.all(A[i, :] == np.inf)]
        idx_lone_trk = [j for j in range(A.shape[1]) if np.all(A[:, j] == np.inf)]
        if not all_assigned:
            left_rows = [i for i in range(A.shape[0]) if i not in idx_lone_det]
            left_cols = [j for j in range(A.shape[1]) if j not in idx_lone_trk]
            A = np.delete(A, idx_lone_det, axis=0)  # returns copy
            if A.size > 0:
                A = np.delete(A, idx_lone_trk, axis=1)  # returns copy
    else:
        idx_lone_det = []
        idx_lone_trk = []
        left_rows = list(range(A.shape[0]))
        left_cols = list(range(A.shape[1]))

    if A.size == 0:
        return OneEdgeBipartiteGraph({}, nrows, ncols, 0)

    # Assign obvious ones
    pass  # TODO

    # Run assignment, if needed
    if algorithm == "JVC":
        large_value = 1e8
        A = np.where(A == np.inf, large_value, A)
        row_ind, col_ind = linear_sum_assignment(A, maximize=False)
    elif algorithm in ["auction"]:
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Map back to indices
    if not all_assigned:
        row_ind = [left_rows[i] for i in row_ind]
        col_ind = [left_cols[i] for i in col_ind]

    # Update assignment
    idx_lone_det.extend(list(set(left_rows) - set(row_ind)))
    idx_lone_trk.extend(list(set(left_cols) - set(col_ind)))
    assignments = [
        (ri, cj)
        for ri, cj in zip(row_ind, col_ind)
        if (ri not in idx_lone_det) and (cj not in idx_lone_trk)
    ]

    # Remove illegal assignments
    if not all_assigned:
        rem_pairs = []
        for ri, cj in assignments:
            if A[left_rows.index(ri), left_cols.index(cj)] == large_value:
                rem_pairs.append((ri, cj))
                idx_lone_det.append(ri)
                idx_lone_trk.append(cj)
        for rem in rem_pairs:
            assignments.remove(rem)

    # Get cost from assignments
    cost = 0
    for assign in assignments:
        cost += A[left_rows.index(assign[0]), left_cols.index(assign[1])]

    # Check assignment
    if not all_assigned:
        for ri, cj in assignments:
            if A[left_rows.index(ri), left_cols.index(cj)] == large_value:
                raise RuntimeError(
                    f"Assignment made illegal choice -- "
                    + f"{A}\n{assignments}\n({ri},{cj}) -- {A[left_rows.index(ri), left_cols.index(cj)]}\n"
                    + f"row: {A[left_rows.index(ri),:]}\ncol: {A[:,left_cols.index(cj)]}"
                )

    # Make assignment
    row_to_col = {a[0]: a[1] for a in assignments}
    return OneEdgeBipartiteGraph(row_to_col, nrows, ncols, cost)


def convert_matrix_to_list(A):
    return [(c,) + np.unravel_index(idx, A.shape) for idx, c in enumerate(np.ravel(A))]


def greedy_assignment(A, threshold):
    """Perform greedy association based on a cost matrix

    A -- cost matrix to minimize
    threshold -- maximum cost threshold to consider
    """
    nrows, ncols = A.shape
    A_list = convert_matrix_to_list(A)
    assigns = {}
    assigns_rev = {}
    nassigns = 0
    cost = 0
    assign_costs = sorted(A_list, reverse=False)
    for row in assign_costs:
        a, i, j = row
        if (nassigns == min(nrows, ncols)) or (a > threshold):
            break
        else:
            if (i not in assigns) and (j not in assigns_rev):
                assigns[i] = j
                assigns_rev[j] = i
                nassigns += 1
                cost += a
    return OneEdgeBipartiteGraph(assigns, nrows, ncols, cost)


def approx_n_best_solutions(
    A: np.ndarray, n: int, algorithm: str = "JVC", verbose: bool = False
):
    """Implementation of Murty's algorithm with a generator

    Only the first two returns are guaranteed to be in the correct order
    """
    m = A.shape[1]
    assign_sol_best = gnn_single_frame_assign(
        A, algorithm=algorithm, all_assigned=False
    )
    n_required_feasible = len(assign_sol_best)  # must not lose assignments
    best_sols = PrioritySet(
        max_size=np.inf, max_heap=True
    )  # making max removes highest first
    best_sols.push(assign_sol_best.cost, assign_sol_best)
    sol_to_constraint_map = {assign_sol_best: ([], [])}
    yield assign_sol_best

    for i_sweep in range(n - 1):
        # -- get the next best as a starting point
        # cost_start, sol_start = best_sols.top()
        cost_start, sol_start = best_sols.n_smallest(n)[i_sweep]
        cons_yes_start, cons_no_start = sol_to_constraint_map[sol_start]
        sol = sol_start.copy()
        cons_yes = cons_yes_start.copy()
        cons_no = cons_no_start.copy()

        # -- all sequential manipulations starting at current spot
        for i_subsweep in range(0, m - len(cons_yes_start)):
            # swap constraints
            if i_subsweep > 0:
                try:
                    cons_yes.append(cons_no.pop())
                except IndexError:
                    break

            for cand_assign in sol.assignment_tuples:
                if (cand_assign not in cons_no) and (cand_assign not in cons_yes):
                    cons_no.append(cand_assign)
                    break
            else:
                continue
                # raise RuntimeError('Could not find suitable next constraint')

            # apply no constraints by adding inf's
            A_new = A.copy()
            for i_no, j_no in cons_no:
                A_new[i_no, j_no] = np.inf

            # apply yes constraints by eliminating rows/cols
            del_rows = [c[0] for c in cons_yes]
            del_cols = [c[1] for c in cons_yes]

            # create mapping to new entries
            rows_new_to_old = [ir for ir in range(A.shape[0]) if ir not in del_rows]
            cols_new_to_old = [ic for ic in range(A.shape[1]) if ic not in del_cols]
            A_new = np.delete(A_new, del_rows, axis=0)  # remove "yes" rows
            A_new = np.delete(A_new, del_cols, axis=1)  # remove "yes" cols

            # run assignment, check feasible
            assign_next = gnn_single_frame_assign(
                A_new, algorithm=algorithm, all_assigned=False
            )
            feasible = False
            if len(assign_next) >= n_required_feasible - len(del_cols):
                feasible = True
                # map indices back
                assignments = [
                    (rows_new_to_old[ri], cols_new_to_old[cj])
                    for ri, cj in assign_next.assignment_tuples
                ]
                assignments.extend(cons_yes)
                cost = sum(A[ri, cj] for ri, cj in assignments)
                row_to_col = {a[0]: a[1] for a in assignments}
                assign_next = OneEdgeBipartiteGraph(
                    row_to_col, A.shape[0], A.shape[1], cost
                )
                # store in set
                best_sols.push(assign_next.cost, assign_next)
                sol_to_constraint_map[assign_next] = (cons_yes.copy(), cons_no.copy())
                yield assign_next
    else:
        pass


def n_best_solutions(
    A: np.ndarray, n: int, algorithm: str = "JVC", verbose: bool = False
):
    """Implementation of Murty's algorithm

    Guaranteed to be in the correct order
    """
    best_sols = PrioritySet(
        max_size=n,
        max_heap=True,
    )  # making max removes highest first
    approx_best_gen = approx_n_best_solutions(A, n, algorithm, verbose)
    for sol in approx_best_gen:
        best_sols.push(sol.cost, sol)
    return [s[1] for s in best_sols.n_smallest(n)]


def pda_single_frame_assign(gate_map, d2_map, S_map, PD, beta, nrows, ncols):
    """
    :gate_map -- {j1:[i1, i2,...], j2} where j is track and i is measurement
    :d2_map -- {}
    """
    assert ncols == len(gate_map)

    # form hypothesis probabilities
    p_hyp = {j: {} for j in gate_map}
    cost = 0
    for j in gate_map:  # tracks
        b = beta * np.sqrt(np.linalg.det(S_map[j]))
        alpha_is = {i: PD * np.exp(-d2_map[j][i] / 2) for i in gate_map[j]}
        sum_alpha_is = sum(alpha_is.values())
        for i in [-1] + gate_map[j]:  # gated msmts for the track
            if i == -1:  # no assignment
                p_hyp[j][i] = b / (b + sum_alpha_is)
            else:  # assignment according to gate map
                p_hyp[j][i] = alpha_is[i] / (b + sum_alpha_is)
                cost += d2_map[j][i] * p_hyp[j][i]

    # create assignment solution
    gate_map_rev = invert_dict_of_list(gate_map)
    assignments = {i: {j: p_hyp[j][i] for j in gate_map_rev[i]} for i in gate_map_rev}
    assign_sol = MultiEdgeBipartiteGraph(assignments, nrows, ncols, cost=cost)
    return assign_sol


def p_H(d2_map, S_map, PD, beta, i, j):
    """Hypothesis probability for JPDA
    NOTE: Can be optimized by pre-computing"""
    if i == -1:
        p = (1 - PD) * beta
    else:
        M = S_map[j].shape[0]
        g = np.exp(-d2_map[j][i] / 2) / (
            ((2 * np.pi) ** (M / 2)) * np.sqrt(np.linalg.det(S_map[j]))
        )
        p = g * PD
    return p


def jpda_single_frame_assign(
    gate_map, d2_map, S_map, PD, beta, nrows, ncols, A=None, method="combinatorial"
):
    """
    :gate_map -- {j1:[i1, i2,...], j2:...} where j is track and i is measurement
    :d2_map -- {}
    """
    p_H_partial = partial(p_H, d2_map, S_map, PD, beta)

    # Get the assignments and their probabilities
    if method == "combinatorial":
        best_sols, probs = _jpda_via_combinatorics(gate_map, p_H_partial, nrows, ncols)
    elif method == "n_best":
        best_sols, probs = _jpda_via_n_best()
    else:
        raise NotImplementedError

    # Make assignment solution
    assignments = {}
    cost = 0
    for sol, p in zip(best_sols, probs):
        for i, j in sol.iterate_over("row").items():
            j = j[0]
            if i not in assignments:
                assignments[i] = {j: p}
            elif j not in assignments[i]:
                assignments[i][j] = p
            else:
                assignments[i][j] += p
            cost += p * d2_map[j][i]
    assign_sol = MultiEdgeBipartiteGraph(assignments, nrows, ncols, cost=cost)
    return assign_sol


def _jpda_via_combinatorics(gate_map, p_H_partial, nrows, ncols):
    """Get jpda probabilities using all combinations"""
    # Get all combinations
    all_gates = [[-1] + g for g in gate_map.values()]  # add -1 for no assignment
    all_combos = itertools.product(*all_gates)

    # Score combination
    best_sols = [
        OneEdgeBipartiteGraph(
            {i: j for i, j in zip(combo, gate_map.keys()) if i != -1},
            nrows,
            ncols,
            cost=0,
        )
        for combo in all_combos
        if len(set(combo).difference([-1])) == len([c for c in combo if c != -1])
    ]
    likelihoods = [
        np.prod(
            [p_H_partial(i, j[0]) for i, j in sol.iterate_over("row").items()]
            + [p_H_partial(-1, j) for j in sol.unassigned_cols]
        )
        for sol in best_sols
    ]
    # Normalize
    sl = sum(likelihoods)
    probs = [l / sl for l in likelihoods]
    return best_sols, probs


def _jpda_via_n_best():
    """Get jpda probabilities using efficient n best algorithm"""
    raise NotImplementedError
    # Get assignments
    p_hyp = {j: {} for j in gate_map}
    cost = 0
    n_gates = [len(g) + 1 for g in gate_map.values()]  # add 1 for no assignment
    n_combos_max = np.prod(n_gates)  # overestimate of the number of combinations
    n_sols = min(n_sols, n_combos_max)

    # Score each assignment with likelihood/normalized probability
    best_sols = n_best_solutions(A, n=n_sols)
    sol_scores = []
    for sol in best_sols:
        sol_scores.append()


def multi_frame_assignment():
    raise NotImplementedError
