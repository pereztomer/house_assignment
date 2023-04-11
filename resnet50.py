import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten


def create_resnet50(input_shape, num_classes=3):
    # Define input tensor
    input_tensor = tf.keras.layers.Input(shape=input_shape)

    # Normalize input tensor
    x = tf.keras.layers.experimental.preprocessing.Normalization(mean=[103.939, 116.779, 123.68], variance=[1, 1, 1])(
        input_tensor)

    # Create ResNet50 base model
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=x)

    # freeze the layers in the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom output layers
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output_tensor = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Build functional model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    # model.summary()
    return model


def main():
    model = create_resnet50((224, 224, 3))
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='preprocessed_data',
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=123,
        validation_split=0.7,
        subset="training")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory='preprocessed_data',
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=123,
        validation_split=0.3,
        subset="validation")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])
    epochs = 25
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)


if __name__ == '__main__':
    main()
