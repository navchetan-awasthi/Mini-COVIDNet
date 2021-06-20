
"""
Mobile-Net model.
"""
from tensorflow.keras.applications import VGG16, NASNetMobile, MobileNet
from tensorflow.keras.layers import (
    AveragePooling2D, Dense, Dropout, Flatten, Input, BatchNormalization, ReLU, PReLU
)
from tensorflow.keras.models import Model


def get_model(
    input_size: tuple = (224, 224, 3),
    hidden_size: int = 128,
    dropout: float = 0.5,
    num_classes: int = 3
):

    # load the VGG16 network, ensuring the head FC layer sets are left off
    baseModel = MobileNet(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=input_size)
    )
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(hidden_size)(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = PReLU()(headModel)
    headModel = Dropout(dropout)(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)

    # place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    return model
