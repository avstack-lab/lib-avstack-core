from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer


from avstack.config import HOOKS
from avstack.metrics import get_instantaneous_metrics


@HOOKS.register_module()
class MetricsHook:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def __call__(
        self,
        tracks_fused: "DataContainer",
        truths: "DataContainer",
        assign_radius: float = 2.0,
        logger=None,
        *args,
        **kwargs,
    ):
        self.instantanteous_metrics = get_instantaneous_metrics(
            tracks=tracks_fused.data,
            truths=truths.data,
            assign_radius=assign_radius,
        )
        if self.verbose:
            if logger is not None:
                logger.info("Ran metrics evaluator")
