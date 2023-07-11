# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-21
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-29
# @Description:
"""
Custom data structures. Some based on the heapq library.
"""
from __future__ import annotations

import heapq
import json
import time
from copy import deepcopy
from functools import total_ordering
from typing import Any, Dict, List

import numpy as np


def is_iterable(item):
    try:
        _ = iter(item)
    except TypeError:
        return False
    else:
        return True


def invert_dict_of_list(d):
    d_new = {}
    for k, v in d.items():
        for x in v:
            d_new.setdefault(x, []).append(k)
    return d_new


def custom_chain(*it):
    for iterab in it:
        yield from iterab


class _Priority:
    """Base class for priority queue-type objects

    max_heap (bool):
        If True, is a max-heap where the "top" item has the "highest" priority,
        if False, is a min-heap where the "top" item has the "lowest" priority.
    """

    def __init__(self, max_size: int, max_heap: bool = False):
        self.max_size = max_size
        self.heap = []
        self.is_max = max_heap
        self.mult = -1 if max_heap else 1
        self.max_priority = -np.inf

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.TYPE} with {len(self)} items of priority: {[self.mult*v[0] for v in sorted(self.heap)]}"

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        return iter(self.heap)

    def __getitem__(self, idx):
        priority, item = sorted(self.heap)[idx]
        return self.mult * priority, item

    def clear(self):
        self.heap = []

    def full(self):
        return (self.max_size is not None) and (len(self) >= self.max_size)

    def empty(self):
        return len(self.heap) == 0

    def top(self):
        # index, _ = min(enumerate(self.heap), key=lambda x: x[1])
        priority, item = self.heap[0]
        return self.mult * priority, item

    def bottom(self):
        index, _ = max(enumerate(self.heap), key=lambda x: x[1])
        priority, item = self.heap[index]
        return self.mult * priority, item

    def n_largest(self, n):
        if self.is_max:
            hsort = sorted(self.heap)[:n]
            return [(self.mult * v[0], v[1]) for v in hsort]
        else:
            return sorted(self.heap, reverse=True)[:n]

    def n_smallest(self, n):
        if self.is_max:
            hsort = sorted(self.heap, reverse=True)[:n]
            return [(self.mult * v[0], v[1]) for v in hsort]
        else:
            return sorted(self.heap)[:n]

    def pop_all_below(self, priority_max, with_priority=False):
        """
        NOTE: THIS IS SLOW FOR MAX HEAP
        """
        all_things = []
        while (
            self.is_max and (not self.empty()) and (self.bottom()[0] <= priority_max)
        ) or (
            (not self.is_max) and (not self.empty()) and (self.top()[0] <= priority_max)
        ):
            all_things.append(self.pop(with_priority=with_priority))
        return all_things

    def pop_all_above(self, priority_min):
        """
        NOTE: THIS IS SLOW FOR MIN HEAP
        """
        all_things = []
        while (
            self.is_max and (not self.empty()) and (self.top()[0] >= priority_min)
        ) or (
            (not self.is_max)
            and (not self.empty())
            and (self.bottom()[0] >= priority_min)
        ):
            all_things.append(self.pop())
        return all_things


class PriorityQueue(_Priority):
    """Priority queue

    Attributes:
        max_size (int):
            Maximum number of elements to be stored in the queue
        max_heap (bool):
            If True, is a max-heap where the "top" item has the "highest" priority,
            if False, is a min-heap where the "top" item has the "lowest" priority.
        empty_returns_none (bool):
            If True, popping when empty returns "None",
            if False, popping when empty causes an error.
    """

    TYPE = "PriorityQueue"

    def __init__(
        self,
        max_size: int = None,
        max_heap: bool = False,
        empty_returns_none: bool = False,
    ):
        super().__init__(max_size, max_heap)
        self.empty_returns_none = empty_returns_none

    def push(self, priority, item):
        """Push element onto heap"""
        if not self.full():
            try:
                heapq.heappush(self.heap, (self.mult * priority, item))
            except TypeError as e:
                # print(priority, item, item.source_identifier)
                # raise e
                pass
        else:
            self.pushpop(priority, item)
        self.max_priority = max(self.max_priority, self.mult * priority)

    def pop(self, with_priority=True):
        """Pop top-priority item (smallest in min-heap)"""
        if self.empty() and self.empty_returns_none:
            if with_priority:
                return None, None
            else:
                return None
        else:
            priority, item = heapq.heappop(self.heap)
            if with_priority:
                return self.mult * priority, item
            else:
                return item

    def pushpop(self, priority, item):
        """Push item then return item"""
        try:
            priority, item = heapq.heappushpop(self.heap, (self.mult * priority, item))
        except TypeError as e:
            # print(priority, item, item.source_identifier)
            # raise e
            pass
        return self.mult * priority, item


class PrioritySet(_Priority):
    """Priority set

    Differs from a PriorityQueue in that duplicate items are not allowed.
    Duplicate priorities for different items are allowed.

    Attributes:
        max_size (int):
            Maximum number of elements to be stored in the queue
        max_heap (bool):
            If True, is a max-heap where the "top" item has the "highest" priority,
            if False, is a min-heap where the "top" item has the "lowest" priority.
        empty_returns_none (bool):
            If True, popping when empty returns "None",
            if False, popping when empty causes an error.
        allow_priority_update (bool):
            If True, adding a duplicate item will update the priority to the new value,
            If False, adding a duplicate entry will do nothing.
    """

    TYPE = "PrioritySet"

    def __init__(
        self,
        max_size: int = None,
        max_heap: bool = False,
        empty_returns_none: bool = False,
        allow_priority_update: bool = False,
    ):
        """Min/Max heap wrapper for priority set"""
        super().__init__(max_size, max_heap)
        self.allow_priority_update = allow_priority_update
        if allow_priority_update:
            raise NotImplementedError
        self.empty_returns_none = empty_returns_none
        self.item_set = set()
        self.item_priority_map = {}

    def push(self, priority, item):
        """Push element onto heap"""
        if not self.full():
            if item not in self.item_set:
                heapq.heappush(self.heap, (self.mult * priority, item))
                self.item_set.add(item)
                self.item_priority_map[item] = self.mult * priority
            else:
                if self.allow_priority_update:
                    raise NotImplementedError
        else:
            self.pushpop(priority, item)

    def pop(self):
        """Pop top-priority item from set (smallest in min-heap)"""
        if self.empty() and self.empty_returns_none:
            return None, None
        else:
            priority, item = heapq.heappop(self.heap)
            self.item_set.remove(item)
            del self.item_priority_map[item]
            return self.mult * priority, item

    def pushpop(self, priority, item):
        """Push item then return item

        NOTE: this may be buggy with duplicate entries
        """
        if item not in self.item_set:
            res_priority, res_item = heapq.heappushpop(
                self.heap, (self.mult * priority, item)
            )
            # swap items in set if we changed
            if res_item != item:
                self.item_set.remove(res_item)
                del self.item_priority_map[res_item]
                self.item_set.add(item)
                self.item_priority_map[item] = self.mult * priority
            return self.mult * res_priority, res_item
        else:
            if self.allow_priority_update:
                raise NotImplementedError


@total_ordering
class BipartiteGraph:
    """Base class for assignment solution

    A bipartite graph is a mapping between two disjoint sets of nodes
    """

    def __init__(self, row_to_col: dict, nrow: int, ncol: int, cost: float):
        """
        :row_to_col: dictionary with assignments and weights, e.g.,
        {1:{2:0.1, 4:0.25}, 3:{1:1.0}}
        """
        self._row_to_col = row_to_col
        self._col_to_row = {}
        for r, cs in row_to_col.items():
            if r >= nrow:
                raise RuntimeError(
                    f"Row index {r} cannot be larger than number of rows {nrow} (0-index)"
                )
            for c, w in cs.items():
                if c >= ncol:
                    raise RuntimeError(
                        f"Column index {c} cannot be larger than number of cols {ncol} (0-index)"
                    )
                if c not in self._col_to_row:
                    self._col_to_row[c] = {}
                self._col_to_row[c][r] = w
        self._idx_row = list(range(nrow))
        self._idx_col = list(range(ncol))
        self.nrow = nrow
        self.ncol = ncol
        self.unassigned_rows = tuple(
            [r for r in self._idx_row if r not in self._row_to_col]
        )
        self.unassigned_cols = tuple(
            [c for c in self._idx_col if c not in self._col_to_row]
        )
        self.cost = cost

        # for hashing...
        self._assign_list = tuple(
            sorted(
                [
                    (r, tuple(sorted([(c, p) for c, p in self._row_to_col[r].items()])))
                    for r in self._row_to_col
                ]
            )
        )

    def iterate_over(self, over):
        if over.lower() in ["col", "cols"]:
            return self._col_to_row
        elif over.lower() in ["row", "rows"]:
            return self._row_to_col
        else:
            raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Solution to assignment problem with cost {self.cost} and:\n   Assignments (r-->c): {self._row_to_col}"
            + f"\n   Lone Rows: {self.unassigned_rows}\n   Lone Cols: {self.unassigned_cols}"
        )

    def __eq__(self, other):
        return self.cost == other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __hash__(self):
        return hash((self.nrow, self.ncol, self.cost, self._assign_list))

    def __len__(self):
        return len(self._row_to_col)

    def same(self, other):
        """Check if two graphs are the same via hashing"""
        if self.__hash__() == other.__hash__():
            return True
        else:
            return False

    def assigns_by_row(self, row):
        return tuple([c for c in self._row_to_col[row]])

    def assigns_by_col(self, col):
        return tuple([r for r in self._col_to_row[col]])

    def assign_probability(self, row, col):
        return self._row_to_col[row][col]


class OneEdgeBipartiteGraph(BipartiteGraph):
    """Data structure permitting only single linkage between rows and columns

    Attributes:
        row_to_col (dict):
            The dictionary mapping indices of rows to indices of columns. Can be
            formatted as {i:j, p:q} or {i:{j: 1.0}, p:{q: 1.0}} where the 1.0
            represents the weight of edge. It must be 1.0 in a one-edge graph.
        nrow (int):
            Number of items in the rows set of nodes.
        ncol (int):
            Number of items in the cols set of nodes
        cost (float):
            Total cost of the assignment matching.
    """

    def __init__(self, row_to_col: dict, nrow: int, ncol: int, cost: float):
        """
        :row_to_col - can either be in the format of dictionary with assignments and weights
        or can be a dictionary without weights which will be converted
        """
        if len(row_to_col) > 0:
            c1 = row_to_col[list(row_to_col.keys())[0]]
            if isinstance(c1, (np.int64, int)):
                r2c = {r: {c: 1.0} for r, c in row_to_col.items()}
            elif (
                isinstance(c1, dict) and len(c1) == 1 and c1[list(c1.keys())[0]] == 1.0
            ):
                r2c = row_to_col  # already in correct format
            else:
                raise NotImplementedError(
                    f"{row_to_col} of incompatible typing, {type(c1)}"
                )
        else:
            r2c = {}
        super().__init__(r2c, nrow, ncol, cost)
        self.assignment_tuples = tuple([(r, list(r2c[r].keys())[0]) for r in r2c])

    def copy(self):
        return OneEdgeBipartiteGraph(self._row_to_col, self.nrow, self.ncol, self.cost)

    def deepcopy(self):
        return OneEdgeBipartiteGraph(
            deepcopy(self._row_to_col), self.nrow, self.ncol, self.cost
        )


class MultiEdgeBipartiteGraph(BipartiteGraph):
    """Data structure permitting assignment of multiple rows to a single column

    Similar to the OneEdgeBipartiteGraph except in this case we allow for multiple
    edges between a "row" node to "column" nodes. In this case, we must specify the
    weights for each edge.

    Attributes:
        row_to_col (dict):
            The dictionary mapping indices of rows to indices of columns. Must be
            formatted as {i:{j:0.75, k:0.25}, p:{q:0.5, r:0.5}} where the weights
            for a given row node must sum to 1.0.
        nrow (int):
            Number of items in the rows set of nodes.
        ncol (int):
            Number of items in the cols set of nodes
        cost (float):
            Total cost of the assignment matching.
    """

    def __init__(self, row_to_col: dict, nrow: int, ncol: int, cost: float):
        """
        :row_to_col - must be in the format of dictionary with assignments and weights
        """
        super().__init__(row_to_col, nrow, ncol, cost)

    def copy(self):
        return MultiEdgeBipartiteGraph(
            self._row_to_col, self.nrow, self.ncol, self.cost
        )

    def deepcopy(self):
        return MultiEdgeBipartiteGraph(
            deepcopy(self._row_to_col), self.nrow, self.ncol, self.cost
        )


class DataManager:
    """Manages data buckets where each data bucket is treated as a PriorityQueue

    Each push input must be something that has a "source identifier" so a
    suitable data bucket can be created. For example, using a SensorData
    derived class will suffice. Each time you add sensor data, it will
    be stored into its own sensor's DataBucket.

    Attributes:
        max_size (int):
            Maximum size for each data bucket.
    """

    TYPE = "DataManager"

    def __init__(self, max_size: int = 10):
        self.data = {}
        self.max_size = max_size

    @property
    def n_buckets(self):
        return len(self.data)

    @property
    def n_data(self):
        return sum([len(b) for b in self.data.values()])

    @property
    def keys(self):
        return list(self.data.keys())

    def __str__(self):
        return f"{self.TYPE} with data: {self.data}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def empty(self):
        return all([d.empty() for d in self.data.values()])

    def data_ids(self):
        return list(self.data.keys())

    def data_names(self):
        return list(self.data.keys())

    def push(self, data):
        if data is not None:
            if isinstance(data, list):
                raise TypeError("Cannot process list in data manager")
            if data.source_identifier not in self.data:
                self.data[data.source_identifier] = DataBucket(
                    data.source_identifier, self.max_size
                )
            self.data[data.source_identifier].push(data)

    def has_data(self, s_ID):
        if (s_ID in self.data) and (len(self.data[s_ID]) > 0):
            return True
        else:
            return False

    def pop(self, s_ID=None):
        try:
            if s_ID is None:
                return {ID: self.data[ID].pop() for ID in self.data}
            else:
                return self.data[s_ID].pop()
        except KeyError as e:
            raise KeyError(f"{self} does not have key {s_ID}, has {self.keys}")

    def top(self, s_ID=None):
        try:
            if s_ID is None:
                return {ID: self.data[ID].top() for ID in self.data}
            else:
                return self.data[s_ID].top()
        except KeyError as e:
            raise KeyError(f"{self} does not have key {s_ID}, has {self.keys}")

    def pop_all_below(self, priority, s_ID=None):
        try:
            if s_ID is None:
                return {ID: self.data[ID].pop_all_below(priority) for ID in self.data}
            else:
                return self.data[s_ID].pop_all_below(priority)
        except KeyError as e:
            raise KeyError(f"{self} does not have key {s_ID}, has {self.keys}")

    def get_highest_earliest_priority(self):
        """Get the highest earliest priority (max of mins) from all the buckets

        Buckets are min-heaps, so the "top" returns the lowest priority. This is fast.
        """
        t_earliest = [bucket.top()[0] for bucket in self.data.values()]
        return max(t_earliest) if len(t_earliest) > 0 else None

    def get_lowest_latest_priority(self):
        """Get the lowest latest priority (min of maxes) from all the buckets

        Buckets are max-heaps, so the "bottom" returns the highest priority. This is slow.
        """
        t_latest = [bucket.bottom()[0] for bucket in self.data.values()]
        return min(t_latest) if len(t_latest) > 0 else None


class DataBucket(PriorityQueue):
    """Manages data elements over time

    A priority queue with some sugar on top

    Attributes:
        source_identifier (string):
            The identifier of this data bucket. Should be a unique string
        max_size (int):
            Maximum number of elements to be stored in the queue
        empty_returns_none (bool):
            If True, popping when empty returns "None",
            if False, popping when empty causes an error.
        max_heap (bool):
            If True, is a max-heap where the "top" item has the "highest" priority,
            if False, is a min-heap where the "top" item has the "lowest" priority.
    """

    TYPE = "DataBucket"

    def __init__(
        self,
        source_identifier: str,
        max_size: int = 10,
        empty_returns_none: bool = True,
        max_heap: bool = False,
    ):
        super().__init__(
            max_size, empty_returns_none=empty_returns_none, max_heap=max_heap
        )
        self.source_identifier = source_identifier

    def push(self, data):
        assert (
            data.source_identifier == self.source_identifier
        ), f"{data.source_identifier} needs to match {self.source_identifier}"
        super(DataBucket, self).push(data.timestamp, data)

    def pop(self, with_priority=False):
        if with_priority:
            return super(DataBucket, self).pop()
        else:
            return super(DataBucket, self).pop()[1]


class DataContainerEncoder(json.JSONEncoder):
    def default(self, o):
        dc_dict = {
            "frame": o.frame,
            "timestamp": o.timestamp,
            "source_identifier": o.source_identifier,
            "data": [d.encode() for d in o.data],
        }
        return {"datacontainer": dc_dict}


class DataContainerDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        self.data_decoder = None

    def object_hook(self, json_object):
        if "datacontainer" in json_object:
            json_object = json_object["datacontainer"]
            return DataContainer(
                frame=json_object["frame"],
                timestamp=json_object["timestamp"],
                data=[
                    json.loads(d, cls=self.data_decoder) for d in json_object["data"]
                ],
                source_identifier=json_object["source_identifier"],
            )
        else:
            return json_object


class DataContainer:
    """Manages data elements at single snapshot in time

    For example, a collection of detection results can be stored
    in a data container. Anything in a data container is assumed (forced)
    to have the same "fundamentals" which includes frame, timestamp, and
    source identifier.

    Attributes:
        frame (int):
            Frame that the data were captured
        timestamp (float):
            Timestamp that the data were captured
        data (iterable):
            The collection of data (e.g., a list of detections)
        source_identifier (string):
            The identifier of the source of data
    """

    TYPE = "DataContainer"

    def __init__(self, frame: int, timestamp: float, data: Any, source_identifier: str):
        self.frame = frame
        self.timestamp = timestamp
        self.data = data
        self.source_identifier = source_identifier

    def __str__(self):
        return f"{len(self.data)} elements at frame {self.frame}, time {self.timestamp}, ID: {self.source_identifier}\n{self.data}"

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        if isinstance(other, DataContainer):
            self._check_fundamentals(other)
            return DataContainer(
                self.frame,
                self.timestamp,
                self.data + other.data,
                self.source_identifier,
            )
        elif isinstance(other, list):
            return DataContainer(
                self.frame, self.timestamp, self.data + other, self.source_identifier
            )
        else:
            raise NotImplementedError(f"Cannot add type {type(other)} to {self.TYPE}")

    def __lt__(self, other):
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        else:
            if len(self) != len(other):
                return len(self) < len(other)
            else:
                return hash(self.source_identifier) < hash(other.source_identifier)

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        assert frame >= 0
        self._frame = frame

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        if not isinstance(timestamp, (int, float, np.int64, np.float64)):
            raise TypeError(
                f"Input timestamp of type {type(timestamp)} is not of an acceptable type"
            )
        self._timestamp = timestamp

    def encode(self):
        return json.dumps(self, cls=DataContainerEncoder)

    def append(self, other):
        self.data.append(other)

    def extend(self, other):
        if isinstance(other, DataContainer):
            self._check_fundamentals(other)
            self.data.extend(other.data)
        elif isinstance(other, list):
            self.data.extend(other)
        else:
            raise NotImplementedError(type(other))

    def _check_fundamentals(self, other):
        if isinstance(other, DataContainer):
            if (self.frame != other.frame) or (self.timestamp != other.timestamp):
                raise RuntimeError(
                    f"Mismatch in frame or timestamp -- ({self.frame}, {self.timestamp}, ({other.frame}, {other.timestamp})"
                )
        else:
            raise NotImplementedError


# -------------------------------
# Data Buffers
# -------------------------------


class BasicDataBuffer(DataManager):
    pass


class DelayManagedDataBuffer(DataManager):
    """Data buffer managing emits by a delay factor"""

    def __init__(
        self, dt_delay: float, max_size: int = 10, method: str = "event-driven"
    ) -> None:
        super().__init__(max_size=max_size)
        self.dt_delay = dt_delay
        self.t_real_for_first_data = None
        self.t_first_data = None
        self.t_last_emit = -np.inf
        self.method = method
        assert method in ["event-driven", "real-time"]

    def clean(self):
        """Remove any elements that are too old"""
        self.pop_all_below(self.t_last_emit)

    def emit_one(self):
        """Only emit a single element per queue that satisfies the delay timing"""
        t_emit = self._get_emit_timing()
        elements = {}
        for key in self:
            if len(self[key]) > 0:
                this_t = self.top(key)[0]
                if this_t <= t_emit:
                    elements[key] = self.pop(key)
        return elements

    def emit_all(self):
        """Emit all element per queue that satisfy the delay timing"""
        t_emit = self._get_emit_timing()
        elements = self.pop_all_below(t_emit)
        # TODO: add in the t_last_emit logic here
        return elements

    def _get_emit_timing(self):
        """Get the timing of the data for emitting

        NOTE: current implementation assumes sensors are time synchronized
        i.e., no relative timing differences between sensor global clocks
        Could change this by returning a dictionary for each key in buffer
        """
        if self.method == "event-driven":
            return self._emit_event_driven()
        elif self.method == "real-time":
            return self._emit_real_time()
        else:
            raise NotImplementedError(self.method)

    def _emit_event_driven(self):
        """Emits based on events - more coarsely, but logic is simpler

        Can run above-real-time (simulation rate)

        Waits until there is a new data element some delay amount of time
        """
        priorities = sorted([item[0] for key in self for item in self[key]])
        if len(priorities) < 2:
            return -np.inf
        else:
            t_emit = -np.inf
            for index in range(len(priorities) - 1):
                if (priorities[-1] - priorities[index]) >= self.dt_delay - 1e-6:
                    t_emit = max(t_emit, priorities[index])
            self.t_last_emit = max(self.t_last_emit, t_emit)
            return t_emit

    def _emit_real_time(self):
        """Emits based on real-time delay - more fine-grained emit, but more complicated

        Constrained to run in a real-time system delay

        Monitors the delay in real-time to determine when to emit
        """
        t_now = time.time()
        t_emit = -np.inf
        if not self.empty():
            # -- initialize time offset and last emit time
            if self.t_real_for_first_data is None:
                self.t_real_for_first_data = t_now
                self.t_first_data = self.top(self.keys[0])[0] - 1e-6

            # -- get emit timing
            dt_real_since_first = t_now - self.t_real_for_first_data
            for priority in sorted([item[0] for key in self for item in self[key]]):
                dt_data_since_first = priority - self.t_first_data
                dt_diff = dt_real_since_first - dt_data_since_first
                if dt_diff >= self.dt_delay - 1e-6:
                    t_emit = max(t_emit, priority)
            self.t_last_emit = max(self.t_last_emit, t_emit)
        return t_emit
