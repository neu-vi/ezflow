Training (and evaluating) a model
=========================================================

**EzFlow** provides a training (and evaluation) pipeline which makes it easy to train, validate and evaluate models
on popular optical flow datasets. This is achieved by using the :class:`Trainer` class packaged with the library. Along with training
and validation, the trainer can also be used to easily resume training from a checkpoint. It supports logging, checkpointing, and training 
recipes like distributed training, learning rate scheduling, gradient clipping, and more. To facilitate the training process, **EzFlow** also
contains dataloders for common datasets such as KITTI, FlyingChairs, and FlyingThings3D, and augmentation techniques for the data. It also 
houses a set of common loss functions and metrics.

Along with a training pipeline, the library also provides functions to evaluate models post training, profile inference of models, and 
for pruning of models. 

Let's understand how to use **EzFlow** to train a model on an optical flow dataset.

To start, we need to create dataloader(s) for the dataset(s) we want to train on. We can use the :class:`DataloaderCreator` class for this. 
The class contains methods which can be invoked to create dataloaders for common datasets. It also allows the user to create a 
consolidated dataloader for multiple datasets. Let's take a look at how to create a dataloader for the **FlyingChairs** dataset for example.

.. code-block:: python

    from ezflow.data import DataloaderCreator

    # Instantiate the dataloader creator

    train_loader_creator = DataloaderCreator(
        batch_size=16,
        num_workers=1,
        pin_memory=True
    )
    val_loader_creator = DataloaderCreator(
        batch_size=16,
        num_workers=1,
        pin_memory=True
    )

    # Add dataset(s) to the dataloader creator

    train_loader_creator.add_flying_chairs(
        root_dir="data/FlyingChairs",
        split="training",
        augment=True,
        aug_params={
            "crop_size": (256, 256),
            "color_aug_params": {
                "aug_prob": 0.3,
                "contrast": 0.5
            },
            "spatial_aug_params": {
                "aug_prob": 0.2,
                "flip": True
            }
        }
    ) 
    val_loader_creator.add_flying_chairs(
        root_dir="data/FlyingChairs",
        split="validation",
        augment=False
    ) 

    # Create the dataloaders

    train_loader = train_loader_creator.get_dataloader()
    val_loader = val_loader_creator.get_dataloader()

Next, let's create a RAFT model for training.

.. code-block:: python

    from ezflow.models import build_model

    model = build_model("RAFT", default=True)

Coming to the trainer itself, we need to provide a training configuration object along with the model and dataloaders. In the training 
configuration, we can specify the training hyperparameters, the optimizer, the loss function, the metrics, the callbacks, and more.

We can use :func:`get_training_cfg` function provided with the library to either create a `YACS <https://github.com/rbgirshick/yacs>`_  
configuration object using parameters specified in a YAML configuration file. **EzFlow** provides a few default training configuration files 
which can be used for this purpose. These files are located in the `configs/trainers` directory in the `root <https://github.com/neu-vig/ezflow>`_ of the library.
To use these files, we need to specify the path of the configuration file relative to `configs/trainers`.
Alternatively, a training configuration object can also be created by specifying a custom YAML configuration file.

To use a configuration file packaged with the library:

.. code-block:: python

    from ezflow.engine import get_training_cfg

    training_cfg = get_training_cfg(cfg_path="base.yaml", custom=False)

To use a custom configuration file:

.. code-block:: python

    training_cfg = get_training_cfg(cfg_path="custom_config.yaml", custom=True)

Parameters of the configuration object can be modified manually if desired. For example, we can change the directory 
where the checkpoints are saved.

.. code-block:: python

    training_cfg.CKPT_DIR = "./checkpoints"

(To view all the parameters present in a configration object, the :func:`.to_dict()` method of the object can be used can be used)

Now that we have a training configuration object, we can create a trainer object.

.. code-block:: python

    from ezflow.engine import Trainer

    trainer = Trainer(
        cfg=training_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )

To train the model, we can invoke the :func:`train` method of the trainer.

.. code-block:: python

    trainer.train(n_epochs=10)

    # The number of epochs can also be specified in the training configuration file, in which 
    # case the n_epochs argument is not required.

The trainer can also be used to resume training from a checkpoint as:

.. code-block:: python

    trainer.resume_training(
        consolidated_ckpt="checkpoints/checkpoint_epoch_10.pth",
        n_epochs=10
    )

The `consolidated_ckpt` here is a checkpoint saved previously by EzFlow's trainer which contains checkpoints for model, 
optimizer, and scheduler states. The method can also be used with individual checkpoints which might not have been saved by EzFlow's trainer.

.. code-block:: python

    trainer.resume_training(
        model_ckpt="checkpoints/model_epoch_10.pth",
        optimizer_ckpt="checkpoints/optimizer_epoch_10.pth",
        scheduler_ckpt="checkpoints/scheduler_epoch_10.pth",
        n_epochs=10,
        start_epoch=10
    )

Similar to the training pipeline, **EzFlow** also provides a set of functions to evaluate and profile inference of models.
Along with evaluating a model's accuracy on a dataset, the evaluation functions can also be calculate the inference time, size
and memory consumtion of the model, and more.

.. code-block:: python

    from ezflow.models import build_model
    from ezflow.engine import eval_model

    # Initialize the model from an existing checkpoint
    model = build_model("RAFT", default=True, weights_path="./checkpoints/model_epoch_10.pth")

    # Evaluate the model on the validation dataset and calculate inference time 

    evaluate_model(
        model=model,
        val_loader=val_loader,
        device="0"
    )


In order to evaluate the performance metrics such as memory consumption of the model, **EzFlow** provides a wrapper `ezflow.engine.Profiler` to initialize the parameters for the PyTorch Profiler. The performance metrics can be viewed using the TensorBoard.

.. code-block:: python

    from ezflow.models import build_model
    from ezflow.engine import eval_model, Profiler

    # Initialize the parameters for the profiler

    profiler = Profiler(
        model_name="RAFT",
        log_dir="./profiler_logs",
        profile_cpu=True,
        profile_cuda=True,
        profile_memory=True,
        record_shapes=True,
        wait=1,
        warmup=1,
        active=1,
        repeat=10
    )


    # Evaluate the model on the validation dataset and 
    # collect performance metrics of the model during inference.

    model = build_model("RAFT", default=True, weights_path="./checkpoints/model_epoch_10.pth")

    evaluate_model(
        model=model,
        val_loader=val_loader,
        profiler=profiler,
        device="0"
    )

Please refer to the API documentation to learn more about the trainer, dataloaders, augmentation techniques, evaluation and 
inference functions, and more. We also provide an example training script in the `tools` directory in the 
`root <https://github.com/neu-vig/ezflow>`_ of the library's GitHub repository.

Do check out the other tutorials to learn how to build a custom model using **EzFlow** and how to use one of the already
implemented models.