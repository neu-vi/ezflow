Using one of the implemented optical flow estimation models
==============================================================================

**EzFlow** contains easy-to-use implementations of a number of eminent models for optical flow estimation. 
`PWC-Net <https://arxiv.org/abs/1709.02371>`_, `RAFT <https://arxiv.org/abs/2003.12039>`_, and 
`VCN <https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html>`_ to name a few.

These models can be accessed with the help of builder functions. For example, to build a **RAFT** model, the following code snippet can be used:

.. code-block:: python

    from ezflow.models import build_model

    model = build_model("RAFT", default=True) 

This snippet will return a **RAFT** model with the default configuration and parameters.  

Let's now talk about how the models are implemented and how they can be accessed using the builder functions.

Each model is a composite of sub-modules like encoders and decoders which are present in the library. Every implementation takes in
a `YACS <https://github.com/rbgirshick/yacs>`_ configuration object (:class:`CfgNode`) as input and returns a model object. This configuration object is 
used to supply the various parameters for the encoder, decoder, and other modules and hence to build the model.

**EzFlow** packages default configurations for each models which have the apporpriate parameters for the respective models. To access these default configurations,
the following function can be used:

.. code-block:: python

    from ezflow.models import get_default_model_cfg

    raft_cfg = get_default_model_cfg("RAFT")

The above mentioned getter function reads the YAML configuration file supplied with the library for a model and returns a :class:`CfgNode` object.
This configuration object can be used to build the model.

In the example provided above about using the builder function to access **RAFT**, under the hood, the getter function is used to fetch the default configuration
object for the model name specified and then it is passed to model class constructor to build the model.
However, this is not the only way to access a models present in **EzFlow**. The builder functions also accept a :class:`CfgNode` object as input and return a model object.

.. code-block:: python

    raft_cfg = get_default_model_cfg("RAFT")
    model = build_model("RAFT", cfg=raft_cfg)

This way you can also make a few modifications to the default configuration if required.
To view all the parameters present in a configration object, the :func:`.to_dict()` method of the object can be used can be used.

.. code-block:: python

    raft_cfg = get_default_model_cfg("RAFT")
    raft_cfg.ENCODER.CORR_RADIUS = 5
    model = build_model("RAFT", cfg=raft_cfg)


Additinally, you can also supply YAML configuration file paths to the builder function. These can further be of two types.
**EzFlow** stores config files in the `configs/models` directory in the `root <https://github.com/neu-vig/ezflow>`_ of the library. Files present in this directory can be accessed by specifying the path to the file 
relative to `configs/models`. For example, to build **RAFT** this way:

.. code-block:: python

    model = build_model("RAFT", cfg_path="raft.yaml")

Furthermore, you can also supply a path to a custom YAML configuration file which you may have created for a model.

.. code-block:: python

    model = build_model("RAFT", cfg_path="my_raft_cfg.yaml", custom_cfg=True)

Lastly, the builder function can also be used to load a model with pretrained weights.

.. code-block:: python

    model = build_model("RAFT", default=True, weights_path="raft_weights.pth")


Along with the above described ways to access models, **EzFlow** also provides a higher level API to use these models for prediction.
This can be done using the :class:`Predictor` class.

.. code-block:: python

    from ezflow.models import Predictor
    from torchvision.transforms import Resize

    predictor = Predictor("RAFT", default=True, 
        model_weights_path="raft_weights.pth", 
        data_transform=Resize((256, 256))
    )
    flow = predictor("img1.png", "img2.png")

Please refer to the API documentation for more details. 
Also, do check out out the other tutorials for details on how to use **EzFlow** to build custom models
and how to train them using the training pipeline provided by the library.
