# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-21
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-29
# @Description:
"""

"""

import heapq
from functools import total_ordering
from copy import copy, deepcopy
import numpy as np


def is_iterable(item):
    try:
        iterator = iter(item)
    except TypeError:
        return False
    else:
        return True


def invert_dict_of_list(d):
    d_new = {}
    for k,v in d.items():
        for x in v:
            d_new.setdefault(x, []).append(k)
    return d_new


def custom_chain(*it):
    for iterab in it:
        yield from iterab


class _Priority():
    def __init__(self, max_size, max_heap):
        self.max_size = max_size
        self.heap = []
        self.is_max = max_heap
        self.mult = -1 if max_heap else 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'{self.TYPE} with {len(self)} items of priority: {[self.mult*v[0] for v in sorted(self.heap)]}'

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        return iter(self.heap)

    def __getitem__(self, idx):
        priority, item = sorted(self.heap)[idx]
        return self.mult*priority, item

    def clear(self):
        self.heap = []

    def full(self):
        return (self.max_size is not None) and (len(self) >= self.max_size)

    def empty(self):
        return len(self.heap) == 0

    def top(self):
        # index, _ = min(enumerate(self.heap), key=lambda x: x[1])
        priority, item = self.heap[0]
        return self.mult*priority, item

    def bottom(self):
        index, _ = max(enumerate(self.heap), key=lambda x: x[1])
        priority, item = self.heap[index]
        return self.mult*priority, item

    def n_largest(self, n):
        if self.is_max:
            hsort = sorted(self.heap)[:n]
            return [(self.mult*v[0], v[1]) for v in hsort]
        else:
            return sorted(self.heap, reverse=True)[:n]

    def n_smallest(self, n):
        if self.is_max:
            hsort = sorted(self.heap, reverse=True)[:n]
            return [(self.mult*v[0], v[1]) for v in hsort]
        else:
            return sorted(self.heap)[:n]

    def pop_all_below(self, priority_max):
        """
        NOTE: THIS IS SLOW FOR MAX HEAP
        """
        all_things = []
        while (self.is_max and (not self.empty()) and (self.bottom()[0] <= priority_max)) or \
                ((not self.is_max) and (not self.empty()) and (self.top()[0] <= priority_max)):
            all_things.append(self.pop())
        return all_things

    def pop_all_above(self, priority_min):
        """
        NOTE: THIS IS SLOW FOR MIN HEAP
        """
        while (self.is_max and (not self.empty()) and (self.top()[0] >= priority_min)) or \
                ((not self.is_max) and (not self.empty()) and (self.bottom()[0] >= priority_min)):
            all_things.append(self.pop())
        return all_things


class PriorityQueue(_Priority):
    TYPE = 'PriorityQueue'

    def __init__(self, max_size=None, max_heap=False, empty_returns_none=False):
        super().__init__(max_size, max_heap)
        self.empty_returns_none = empty_returns_none

    def push(self, priority, item):
        """Push element onto heap"""
        if not self.full():
            try:
                heapq.heappush(self.heap, (self.mult*priority, item))
            except TypeError as e:
                # print(priority, item, item.source_identifier)
                # raise e
                pass
        else:
            self.pushpop(priority, item)

    def pop(self):
        """Pop top-priority item (smallest in min-heap)"""
        if self.empty() and self.empty_returns_none:
            return None, None
        else:
            priority, item = heapq.heappop(self.heap)
            return self.mult*priority, item

    def pushpop(self, priority, item):
        """Push item then return item"""
        try:
            priority, item = heapq.heappushpop(self.heap, (self.mult*priority, item))
        except TypeError as e:
            # print(priority, item, item.source_identifier)
            # raise e
            pass
        return self.mult*priority, item


class PrioritySet(_Priority):
    TYPE = 'PrioritySet'

    def __init__(self, max_size=None, max_heap=False, allow_priority_update=False):
        """Min/Max heap wrapper for priority set"""
        super().__init__(max_size, max_heap)
        self.allow_priority_update = allow_priority_update
        if allow_priority_update:
            raise NotImplementedError
        self.item_set = set()
        self.item_priority_map = {}

    def push(self, priority, item):
        """Push element onto heap"""
        if not self.full():
            if item not in self.item_set:
                heapq.heappush(self.heap, (self.mult*priority, item))
                self.item_set.add(item)
                self.item_priority_map[item] = self.mult * priority
            else:
                if self.allow_priority_update:
                    raise NotImplementedError
        else:
            self.pushpop(priority, item)

    def pop(self):
        """Pop top-priority item from set (smallest in min-heap)"""
        priority, item = heapq.heappop(self.heap)
        self.item_set.remove(item)
        del self.item_priority_map[item]
        return self.mult*priority, item

    def pushpop(self, priority, item):
        """Push item then return item

        NOTE: this may be buggy with duplicate entries
        """
        if item not in self.item_set:
            res_priority, res_item = heapq.heappushpop(self.heap, (self.mult*priority, item))
            # swap items in set if we changed
            if res_item != item:
                self.item_set.remove(res_item)
                del self.item_priority_map[res_item]
                self.item_set.add(item)
                self.item_priority_map[item] = self.mult * priority
            return self.mult*res_priority, res_item
        else:
            if self.allow_priority_update:
                raise NotImplementedError


@total_ordering
class BipartiteGraph():
    """Base class for assignment solution"""
    def __init__(self, row_to_col: dict, nrow: int, ncol: int, cost: float):
        """
        :row_to_col: dictionary with assignments and weights, e.g.,
        {1:{2:0.1, 4:0.25}, 3:{1:1.0}}
        """
        self._row_to_col = row_to_col
        self._col_to_row = {}
        for r, cs in row_to_col.items():
            if r >= nrow:
                raise RuntimeError(f'Row index {r} cannot be larger than number of rows {nrow} (0-index)')
            for c, w in cs.items():
                if c >= ncol:
                    raise RuntimeError(f'Column index {c} cannot be larger than number of cols {ncol} (0-index)')
                if c not in self._col_to_row:
                    self._col_to_row[c] = {}
                self._col_to_row[c][r] = w
        self._idx_row = list(range(nrow))
        self._idx_col = list(range(ncol))
        self.nrow = nrow
        self.ncol = ncol
        self.unassigned_rows = tuple([r for r in self._idx_row if r not in self._row_to_col])
        self.unassigned_cols = tuple([c for c in self._idx_col if c not in self._col_to_row])
        self.cost = cost

        # for hashing...
        self._assign_list = tuple(sorted([(r, tuple(sorted([(c, p) \
            for c, p in self._row_to_col[r].items()]))) for r in self._row_to_col]))

    def iterate_over(self, over):
        if over.lower() in ['col', 'cols']:
            return self._col_to_row
        elif over.lower() in ['row', 'rows']:
            return self._row_to_col
        else:
            raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Solution to assignment problem with cost {self.cost} and:\n   Assignments (r-->c): {self._row_to_col}' + \
               f'\n   Lone Rows: {self.unassigned_rows}\n   Lone Cols: {self.unassigned_cols}'

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


class MultiEdgeBipartiteGraph(BipartiteGraph):
    """Data structure permitting assignment of multiple rows to a single column"""
    def __init__(self, row_to_col:dict, nrow:int, ncol:int, cost:float):
        """
        :row_to_col - must be in the format of dictionary with assignments and weights
        """
        super().__init__(row_to_col, nrow, ncol, cost)

    def copy(self):
        return MultiEdgeBipartiteGraph(self._row_to_col, self.nrow, self.ncol, self.cost)

    def deepcopy(self):
        return MultiEdgeBipartiteGraph(deepcopy(self._row_to_col), self.nrow, self.ncol, self.cost)


class OneEdgeBipartiteGraph(BipartiteGraph):
    """Data structure permitting only single linkage between rows and columns"""
    def __init__(self, row_to_col:dict, nrow:int, ncol:int, cost:float):
        """
        :row_to_col - can either be in the format of dictionary with assignments and weights
        or can be a dictionary without weights which will be converted
        """
        if len(row_to_col) > 0:
            c1 = row_to_col[list(row_to_col.keys())[0]]
            if isinstance(c1, (np.int64, int)):
                r2c = {r:{c:1.0} for r, c in row_to_col.items()}
            elif isinstance(c1, dict) and len(c1) == 1 and c1[list(c1.keys())[0]] == 1.0:
                r2c = row_to_col  # already in correct format
            else:
                raise NotImplementedError(f'{row_to_col} of incompatible typing, {type(c1)}')
        else:
            r2c = {}
        super().__init__(r2c, nrow, ncol, cost)
        self.assignment_tuples = tuple([(r,list(r2c[r].keys())[0]) for r in r2c])

    def copy(self):
        return OneEdgeBipartiteGraph(self._row_to_col, self.nrow, self.ncol, self.cost)

    def deepcopy(self):
        return OneEdgeBipartiteGraph(deepcopy(self._row_to_col), self.nrow, self.ncol, self.cost)


class DataManager():
    """Manages data buckets, generally"""
    TYPE = 'DataManager'

    def __init__(self, max_size=10):
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
        return f'{self.TYPE} with data: {self.data}'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.data)

    def empty(self):
        return all([d.empty() for d in self.data.values()])

    def data_ids(self):
        return list(self.data.keys())

    def data_names(self):
        return list(self.data.keys())

    def push(self, data):
        if (data is not None):
            if isinstance(data, list):
                raise TypeError('Cannot process list in data manager')
            if data.source_identifier not in self.data:
                self.data[data.source_identifier] = DataBucket(data.source_identifier, self.max_size)
            self.data[data.source_identifier].push(data)

    def has_data(self, s_ID):
        if (s_ID in self.data) and (len(self.data[s_ID]) > 0):
            return True
        else:
            return False

    def pop(self, s_ID=None):
        try:
            if s_ID is None:
                return {ID:self.data[ID].pop() for ID in self.data}
            else:
                return self.data[s_ID].pop()
        except KeyError as e:
            raise KeyError(f'{self} does not have key {s_ID}, has {self.keys}')


class DataBucket(PriorityQueue):
    """Manages data elements over time"""
    TYPE = 'DataBucket'

    def __init__(self, source_identifier, max_size=10, empty_returns_none=True):
        super().__init__(max_size, empty_returns_none=empty_returns_none)
        self.source_identifier = source_identifier

    def push(self, data):
        assert data.source_identifier == self.source_identifier, f'{data.source_identifier} needs to match {self.source_identifier}'
        super(DataBucket, self).push(data.timestamp, data)

    def pop(self):
        return super(DataBucket, self).pop()[1]

    # def pushpop(self, priority, item):
    #     return super(DataBucket, self).pushpop(data.timestamp, data)[1]


class DataContainer():
    """Manages data elements at single snapshot in time"""
    TYPE = 'DataContainer'

    def __init__(self, frame, timestamp, data, source_identifier):
        self.frame = frame
        self.timestamp = timestamp
        self.data = data
        self.source_identifier = source_identifier

    def __str__(self):
        return f'{len(self.data)} elements at frame {self.frame}, time {self.timestamp}\n{self.data}'

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
            return DataContainer(self.frame, self.timestamp, self.data + other.data, self.source_identifier)
        elif isinstance(other, list):
            return DataContainer(self.frame, self.timestamp, self.data + other, self.source_identifier)
        else:
            raise NotImplementedError(f'Cannot add type {type(other)} to {self.TYPE}')

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
            raise TypeError(f'Input timestamp of type {type(timestamp)} is not of an acceptable type')
        self._timestamp = timestamp

    def append(self, other):
        self.data.append(other)

    def extend(self, other):
        if isinstance(other, DataContainer):
            self._check_fundamentals(other)
            self.data.extend(other.data)
        elif isinstance(other, list):
            self.data.extend(other)
        else:
            raise NotImplementedError(type(other_array))

    def _check_fundamentals(self, other):
        if isinstance(other, DataContainer):
            if (self.frame != other.frame) or (self.timestamp != other.timestamp):
                raise RuntimeError(f'Mismatch in frame or timestamp -- ({self.frame}, {self.timestamp}, ({other.frame}, {other.timestamp})')
        else:
            raise NotImplementedError

    def to_file(self, file, form='kitti'):
        det_str = '\n'.join([det.format_as(form) for det in self.data])
        with open(file, 'w') as f:
            f.write(det_str)
