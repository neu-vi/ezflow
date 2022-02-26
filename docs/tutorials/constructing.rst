Constructing a custom model
=========================================================

**EzFlow** has a modular design, with an aim to make it easy to build custom optical flow estimation models using
component modules such as encoders, similarity functions, and decoders. Popular models for optical flow estimation like
`PWC-Net <https://arxiv.org/abs/1709.02371>`_, `RAFT <https://arxiv.org/abs/2003.12039>`_, and 
`VCN <https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html>`_ have been implemented in the library
by breaking them down into the above mentioned components. **EzFlow** also supports a registry mechanism to allow users to
register modules and use them in custom models seamlessly by specifying the module names in model configurations.

If one does not desire to make use of the registry mechanism, they can directly import the modules from the library and use them as 
they would normally use any other piece of modular code. For example, to use the pyramid feature encoder present in **PWC-Net**:

.. code-block:: python

    from ezflow.encoder import PyramidEncoder

    encoder = PyramidEncoder(in_channels=3, config=[16, 64, 256])
    features = encoder(img1, img2)

Or, to use the learnable matching function present in **DICL**, one can do the following:

.. code-block:: python

    from ezflow.similarity import LearnableMatchingCost

    cost_fn = LearnableMatching(max_u=3, max_v=3)
    cost = cost_fn(features1, features2)

(Please refer to the documentation of the individual modules for more details)

Alternatively, the modules can also be accessed using builder functions which work with the registry mechanism. For example, to access
the pyramid feature encoder from the above example, the following code snippet can be used:

.. code-block:: python

    from ezflow.encoder import build_encoder

    encoder = build_encoder(name="PyramidEncoder", in_channels=3, config=[16, 64, 256])

This function uses the encoder name specified to fetch it from an encoder registry and constructs it using the parameters specified.

While the above examples highlight different ways of accessing component modules present in **EzFlow**, arguably the biggest utility 
of the library is its ability to make it easy to build custom models using the aforementioned modules, registry mechanism, 
builder functions, and configuration system.
To understand how this works, consider the following scenario. Suppose you read the `FlowNet <https://arxiv.org/abs/1504.06852>`_ and
`RAFT <https://arxiv.org/abs/2003.12039>`_ papers and thought it would be interesting to try an ablation analysis of using the residual encoder
present in **RAFT** with the rest of the **FlowNetSimple** architecture. Let's discuss how this can be achieved with **EzFlow**.

A simple way without the registry system and builder functions would be to directly import the modules from the library and then make 
use of them. For example:

.. code-block:: python

    from ezflow.encoder import BasicEncoder # RAFT encoder
    from ezflow.decoder import FlowNetConvDecoder # FlowNetS decoder
    from torch import nn

    # Construct model class using the imported modules

    class MyModel(nn.Module):
        def __init__(self, 
                    encoder_config=[32, 64, 96], 
                    decoder_config=[512, 256, 128, 64]
        ):
            super(MyModel, self).__init__()

            self.encoder = BasicEncoder(in_channels=3, layer_config=encoder_config)
            self.decoder = FlowNetConvDecoder(in_channels=encoder_config[-1], config=decoder_config)

        def forward(self, img1, img2):

            x = torch.cat([img1, img2], axis=1)
            features = self.encoder(x)
            flow_preds = self.decoder(features)
            flow_preds.reverse()

            return flow_preds

Now, let's see how this can be achieved with the registry system and builder functions. First, we need to write a model skeleton 
class which takes in a configuration object and register it to a model registry as shown below:

.. code-block:: python

    from ezflow.decoder import build_encoder
    from ezflow.encoder import build_encoder
    from ezflow.models import MODEL_REGISTRY
    from torch import nn

    class MyModel(nn.Module):
        def __init__(self, cfg):
            super(MyModel, self).__init__()

            self.encoder = build_encoder(cfg.ENCODER)
            self.decoder = build_decoder(cfg.DECODER)

        def forward(self, img1, img2):

            x = torch.cat([img1, img2], axis=1)
            features = self.encoder(x)
            flow_preds = self.decoder(features)
            flow_preds.reverse()

            return flow_preds

Notice that we have used configuration groups in the configuration object to build the encoder and decoder. Keeping this in mind,
we now need to write a suitable YAML configuration file which specifies the encoder and decoder configuration groups.

.. code-block:: yaml

    NAME: MyModel
    ENCODER:
        NAME: ResidualEncoder
        IN_CHANNELS: 3
        OUT_CHANNELS: 256
        LAYER_CONFIG: [32, 64, 96]
        NORM: instance
        P_DROPOUT: 0.0
        INTERMEDIATE_FEATURES: True
    DECODER:
        NAME: FlowNetConvDecoder
        IN_CHANNELS: 1024
        CONFIG: [512, 256, 128, 64]

The model can now be built using the builder function.

.. code-block:: python

    from ezflow.models import build_model

    model = build_model(name="MyModel", cfg_path="MyModel.yaml", custom_cfg=True)
    flow = model(img1, img2)

This whole system can be used to easily mix and match different components. For example, if you wish to use 
the pyramid feature encoder from **PWC-Net**, you simply need modify the encoder configuration group in the configuration file.

.. code-block:: yaml

    NAME: MyModel
    ENCODER:
        NAME: PyramidEncoder
        IN_CHANNELS: 3
        CONFIG: [16, 32, 64, 96, 128, 196]
    DECODER:
        NAME: FlowNetConvDecoder
        IN_CHANNELS: 1024
        CONFIG: [512, 256, 128, 64]

This way one can easily experiment with different model configurations and easily switch between different components.

One can also register their own moduler and use to build custom models. For example, suppose you want to have a custom feature encoder.
You need to perform the following steps to register it to the encoder registry and make it configurable. 

.. code-block:: python

    from ezflow.config import configurable
    from ezflow.encoder import ENCODER_REGISTRY
    from torch import nn

    @ENCODER_REGISTRY.register()
    class MyEncoder(nn.Module):
        @configurable
        def __init__(self, param1, param2, param3):
            super(MyEncoder, self).__init__()

            # ...

        @classmethod
        def from_config(cls, cfg):
            return {
                "param1": cfg.PARAM1,
                "param2": cfg.PARAM2,
                "param3": cfg.PARAM3
            }

        def forward(self, x):

            # ...

The YAML configuration file can now be written as follows:

.. code-block:: yaml

    NAME: MyModel
    ENCODER:
        NAME: MyEncoder
        PARAM1: <param1>
        PARAM2: <param2>
        PARAM3: <param3>
    DECODER:
        NAME: FlowNetConvDecoder
        IN_CHANNELS: 1024
        CONFIG: [512, 256, 128, 64]

The model can now be similarly built using the builder function as described above.

Do check out the other tutorials to understand how to train models using **EzFlow's** training pipeline 
and how to use already implemented models. Please refer to the API documentation for more details.
