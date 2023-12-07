from ezflow.data import DataloaderCreator
from ezflow.engine import eval_model
from ezflow.models import build_model

if __name__ == "__main__":
    model = build_model(
        "DCVNet", weights_path="./pretrained_weights/dcvnet_sceneflow_step800k.pth"
    )

    dataloader_creator = DataloaderCreator(
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )
    dataloader_creator.add_Kitti(
        root_dir="./Datasets/KITTI2015/",
        split="training",
        crop=True,
        crop_type="center",
        crop_size=[370, 1224],
        norm_params={
            "use": True,
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
        },
    )

    kitti_data_loader = dataloader_creator.get_dataloader()
    eval_model(
        model,
        kitti_data_loader,
        device="0",
        pad_divisor=8,
        flow_scale=1.0,
    )

    print("Evaluation Complete!!")
