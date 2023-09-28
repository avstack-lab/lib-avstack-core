from avstack.modules.perception.object2dfv import MMDetObjectDetector2D


class MMInstanceSegmentation(MMDetObjectDetector2D):
    MODE = "instance_segmentation"

    def __init__(
        self,
        model="cascade-mask-rcnn",
        dataset="nuimages",
        deploy=False,
        threshold=None,
        gpu=0,
        epoch="latest",
        deploy_runtime="tensorrt",
        **kwargs,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            deploy=deploy,
            deploy_runtime=deploy_runtime,
            threshold=threshold,
            gpu=gpu,
            epoch=epoch,
            **kwargs,
        )
        
    def initialize(self):
        # This is very hacky....
        from mmdet3d.apis import init_model
        from mmdet3d.utils import register_all_modules
        from mmdet.apis import inference_detector

        register_all_modules(init_default_scope=False)
        self.inference_detector = inference_detector
        self.model = init_model(self.mod_path, self.chk_path, device=f"cuda:{self.gpu}")

    @staticmethod
    def parse_mm_model_from_checkpoint(model, dataset, epoch):
        input_data = "camera"
        label_dataset_override = dataset
        epoch_str = "latest" if epoch == "latest" else "epoch_{}".format(epoch)
        if model == "cascade-mask-rcnn":
            if dataset in ["coco"]:
                threshold = 0.7
                config_file = "configs/cascade_rcnn/cascade-mask-rcnn_r50-caffe_fpn_1x_coco.py"
                checkpoint_file = "checkpoints/coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
                # label_dataset_override = "kitti"
            elif dataset in [
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
