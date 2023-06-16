# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-03
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-26
# @Description:


from . import pools  # import to apply patch to multiprocess pool
from .other import check_xor_for_none, IterationMonitor
from .stats import mean_confidence_interval