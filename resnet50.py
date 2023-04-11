import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten


def create_resnet_classifier(input_shape):
    # load the ResNet50 model pre-trained on ImageNet
    resnet50 = tf.keras.applications.ResNet50(
        include_top=False,  # exclude the top layer of the model
        weights="imagenet",  # use pre-trained ImageNet weights
        input_shape=input_shape  # input shape of the model
    )

    # freeze the layers in the pre-trained model
    for layer in resnet50.layers:
        layer.trainable = False

    # add additional layers for your specific task
    x = resnet50.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(3, activation="softmax")(x)

    # create the final model
    model = tf.keras.models.Model(inputs=resnet50.input, outputs=x)
    # model.summary()
    return model


def main():
    model = create_resnet_classifier((576, 576, 3))
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='preprocessed_data',
        color_mode='rgb',
        batch_size=32,
        image_size=(576, 576),
        shuffle=True,
        seed=123,
        validation_split=0.8,
        subset="training")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory='preprocessed_data',
        color_mode='rgb',
        batch_size=32,
        image_size=(576, 576),
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset="validation")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
    epochs = 25
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)


if __name__ == '__main__':
    main()
