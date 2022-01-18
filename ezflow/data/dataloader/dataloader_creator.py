from torch.utils.data.dataloader import DataLoader

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
    """

    def __init__(
        self,
        batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        init_seed=False,
        append_valid_mask=False,
        is_prediction=False,
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

    def add_flying_chairs(self, root_dir, split="training", augment=True, **kwargs):
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
                **kwargs
            )
        )

    def add_flying_things3d(
        self,
        root_dir,
        split="training",
        dstype="frames_cleanpass",
        augment=True,
        **kwargs
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
                **kwargs
            )
        )

    def add_flying_things3d_subset(
        self, root_dir, split="training", augment=True, **kwargs
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
                **kwargs
            )
        )

    def add_monkaa(self, root_dir, augment=True, **kwargs):
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
                **kwargs
            )
        )

    def add_driving(self, root_dir, augment=True, **kwargs):
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
                **kwargs
            )
        )

    def add_sceneflow(self, root_dir, augment=True, **kwargs):
        """
        Adds Flying Things 3D, Driving and Monkaa datasets to the DataloaderCreator object.

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
        self.add_flying_things3d(root_dir=root_dir + "/FlyingThings3D", augment=augment)
        self.add_monkaa(root_dir=root_dir + "/Monkaa", augment=augment)
        self.add_driving(root_dir=root_dir + "/Driving", augment=augment)

    def add_mpi_sintel(
        self, root_dir, split="training", dstype="clean", augment=True, **kwargs
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
                **kwargs
            )
        )

    def add_kitti(self, root_dir, split="training", augment=True, **kwargs):
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
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs
            )
        )

    def add_hd1k(self, root_dir, augment=True, **kwargs):
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
            Hd1k(
                root_dir,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs
            )
        )

    def add_autoflow(self, root_dir, augment=True, **kwargs):
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
                **kwargs
            )
        )

    def get_dataloader(self):
        """
        Gets the Dataloader for the added datasets.

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

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

        print("Total image pairs: %d" % len(dataset))
        return data_loader
