

from avstack.modules.perception.object2dfv import MMDetObjectDetector2D


class MMInstanceSegmentation(MMDetObjectDetector2D):
    MODE = "instance_segmentation"
    
    def __init__(self, model="cascade-mask-rcnn", dataset="nuimages", threshold=None, gpu=0, epoch="latest", **kwargs):
        super().__init__(model, dataset, threshold, gpu, epoch, **kwargs)

    def initialize(self):
        # This is very hacky....
        from mmdet3d.utils import register_all_modules
        from mmdet3d.apis import init_model
        from mmdet.apis import inference_detector
        
        register_all_modules(init_default_scope=False)
        self.inference_detector = inference_detector
        self.model = init_model(self.mod_path, self.chk_path, device=f"cuda:{self.gpu}")

    @staticmethod
    def parse_mm_model(model, dataset, epoch):
        input_data = "camera"
        label_dataset_override = dataset
        epoch_str = "latest" if epoch == "latest" else "epoch_{}".format(epoch)
        if model == "cascade-mask-rcnn":
            if dataset in [
                "nuimages",
                "carla",
                "kitti",
                "nuscenes",
            ]:  # TODO eventually separate these
                threshold = 0.7
                config_file = (
                    "configs/nuimages/cascade-mask-rcnn_r50_fpn_coco-20e-1x_nuim.py"
                )
                checkpoint_file = "checkpoints/nuimages/cascade_mask_rcnn_r50_fpn_coco-20e_1x_nuim_20201009_124158-ad0540e3.pth"
                label_dataset_override = "nuimages"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        else:
            raise NotImplementedError

        return (
            threshold,
            config_file,
            checkpoint_file,
            input_data,
            label_dataset_override,
        )