from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..dataset import *


class DataloaderCreator:
    """
    A class to configure a data loader for optical flow datasets.
    Multiple datasets can be added to configure a data loader for
    training and validation.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch to load
    pin_memory : bool, default : False
        If True, the data loader will copy Tensors into CUDA pinned memory before returning them
    shuffle : bool, default : True
        If True, data is reshuffled at every epoch
    num_workers : int, default : 4
        Number of subprocesses to use for data loading
    drop_last : bool, default : True
        If True, the last incomplete batch is dropped
    init_seed : bool, default : False
        If True, sets random seed to worker
    append_valid_mask : bool, default :  False
        If True, appends the valid flow mask to the original flow mask at dim=0
    is_prediction : bool, default : False
        If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    distributed : bool, default : False
        If True, initializes DistributedSampler for Distributed Training
    world_size : int, default : None
        The total number of GPU devices per node for Distributed Training
    """

    def __init__(
        self,
        batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        init_seed=False,
        append_valid_mask=False,
        is_prediction=False,
        distributed=False,
        world_size=1,
    ):
        self.dataset_list = []
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.init_seed = init_seed
        self.append_valid_mask = append_valid_mask
        self.is_prediction = is_prediction

        self.distributed = False
        self.world_size = 1

        if distributed:
            assert (
                world_size > 1
            ), "world_size must be greater than 1 to perform distributed training"

            self.distributed = distributed
            self.world_size = world_size

    def add_FlyingChairs(self, root_dir, split="training", augment=False, **kwargs):
        """
        Adds the Flying Chairs dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the flying chairs dataset
        split : str, default : "training"
            specify the training or validation split
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """

        self.dataset_list.append(
            FlyingChairs(
                root_dir,
                split=split,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_FlyingThings3D(
        self,
        root_dir,
        split="training",
        dstype="frames_cleanpass",
        augment=False,
        **kwargs,
    ):
        """
        Adds the Flying Things 3D dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the flying things 3D dataset in SceneFlow
        split : str, default : "training"
            specify the training or validation split
        dstype : str, default : "frames_cleanpass"
            specify dataset type
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """

        self.dataset_list.append(
            FlyingThings3D(
                root_dir,
                split=split,
                dstype=dstype,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_FlyingThings3DSubset(
        self, root_dir, split="training", augment=False, **kwargs
    ):
        """
        Adds the Flying Things 3D Subset dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the flying things 3D Subset dataset in SceneFlow
        split : str, default : "training"
            specify the training or validation split
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """

        self.dataset_list.append(
            FlyingThings3DSubset(
                root_dir,
                split=split,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_Monkaa(self, root_dir, augment=False, **kwargs):
        """
        Adds the Monkaa dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the Monkaa dataset in SceneFlow
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """

        self.dataset_list.append(
            Monkaa(
                root_dir,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_Driving(self, root_dir, augment=False, **kwargs):
        """
        Adds the Driving dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the Driving dataset in SceneFlow
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """

        self.dataset_list.append(
            Driving(
                root_dir,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_SceneFlow(self, root_dir, augment=False, **kwargs):
        """
        Adds FlyingThings3D, Driving and Monkaa datasets to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the SceneFlow dataset
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """
        self.add_FlyingThings3D(
            root_dir=root_dir + "/FlyingThings3D", augment=augment, **kwargs
        )
        self.add_Monkaa(root_dir=root_dir + "/Monkaa", augment=augment, **kwargs)
        self.add_Driving(root_dir=root_dir + "/Driving", augment=augment, **kwargs)

    def add_MPISintel(
        self, root_dir, split="training", dstype="clean", augment=False, **kwargs
    ):
        """
        Adds the MPI Sintel dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the MPI Sintel dataset
        split : str, default : "training"
            specify the training or validation split
        dstype : str, default : "clean"
            specify dataset type
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """
        self.dataset_list.append(
            MPISintel(
                root_dir,
                split=split,
                dstype=dstype,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_Kitti(self, root_dir, split="training", augment=False, **kwargs):
        """
        Adds the KITTI dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the MPI Sintel dataset
        split : str, default : "training"
            specify the training or validation split
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """

        self.dataset_list.append(
            Kitti(
                root_dir,
                split=split,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=True,
                augment=augment,
                **kwargs,
            )
        )

    def add_HD1K(self, root_dir, augment=False, **kwargs):
        """
        Adds the HD1K dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the MPI Sintel dataset
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """
        self.dataset_list.append(
            HD1K(
                root_dir,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_AutoFlow(self, root_dir, augment=False, **kwargs):
        """
        Adds the AutoFLow dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the Monkaa dataset in SceneFlow
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """

        self.dataset_list.append(
            AutoFlow(
                root_dir,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_Kubric(self, root_dir, split="training", augment=False, **kwargs):
        """
        Adds the Kubric dataset to the DataloaderCreator object.

        Parameters
        ----------
        root_dir : str
            path of the root directory for the Monkaa dataset in SceneFlow
        augment : bool, default : True
            If True, applies data augmentation
        **kwargs
            Arbitrary keyword arguments for augmentation
            specifying crop_size and the probability of
            color, eraser and spatial transformation
        """
        self.dataset_list.append(
            Kubric(
                root_dir,
                split=split,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )

    def add_dataset(self, dataset):
        """
        Add an optical flow dataset to the DataloaderCreator object.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            the optical flow dataset
        """
        assert dataset is not None and isinstance(
            dataset, BaseDataset
        ), "Invalid dataset type."
        self.dataset_list.append(dataset)

    def get_dataloader(self, rank=0):
        """
        Gets the Dataloader for the added datasets.

        Params
        ------
        rank : int, default : 0
            The process id within a group for Distributed Training

        Returns
        -------
        torch.utils.data.DataLoader
            PyTorch DataLoader object
        """
        assert len(self.dataset_list) != 0, "No datasets were added"

        dataset = self.dataset_list[0]

        if len(self.dataset_list) > 1:
            for i in range(len(self.dataset_list) - 1):
                dataset += self.dataset_list[i + 1]

        if self.distributed:

            sampler = DistributedSampler(
                dataset,
                rank=rank,
                num_replicas=self.world_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size // self.world_size,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
            )

        total_samples = len(data_loader) * (self.batch_size // self.world_size)
        print(
            f"Total image pairs loaded: {total_samples}/{len(dataset)} in device: {rank}"
        )

        return data_loader
