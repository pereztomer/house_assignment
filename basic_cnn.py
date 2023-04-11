import tensorflow as tf

def basic_cnn(input_shape: tuple):
    """
    creating a basic cnn network, which takes as an input a greyscale image and output a classification
    :param input_shape: a tuple
    :return: tf model
    """
    inputs = tf.keras.Input(shape=input_shape)
    # modify the first Conv2D layer for grayscale images
    x = tf.expand_dims(inputs, axis=-1)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', activation='relu',
                               input_shape=input_shape + (1,))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=5, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=96, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=3, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.summary()
    return model


def main():
    model = basic_cnn(input_shape=(224, 224))
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='extended_ds',
        color_mode='grayscale',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=123,
        validation_split=0.7,
        subset="training")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory='extended_ds',
        color_mode='grayscale',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=123,
        validation_split=0.3,
        subset="validation")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
    epochs = 25
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)


if __name__ == '__main__':
    main()
