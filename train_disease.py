# src/training/train_disease.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

def train_disease_cnn(
    src_dir: str,
    model_out: str,
    img_size=(224, 224),
    batch_size=32,
    epochs=20,
    learning_rate=1e-4
):
    # 1. Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
    )
    train_gen = train_datagen.flow_from_directory(
        src_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        src_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_gen.class_indices)
    print(f"Found {train_gen.samples} training images, {val_gen.samples} validation images across {num_classes} classes")

    # 2. Build a simple CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*img_size, 3)),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 3. Callbacks
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_out,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    earlystop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # 4. Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, earlystop]
    )

    print(f"Training complete. Best model saved to {model_out}")

if __name__ == "__main__":
    # Adjust this path to where your processed images live
    img_dir   = os.path.join(
        os.path.dirname(__file__),
        '../../data/processed/disease_images_224'
    )
    output_h5 = os.path.join(
        os.path.dirname(__file__),
        '../../models/disease_cnn.h5'
    )

    train_disease_cnn(
        src_dir=img_dir,
        model_out=output_h5,
        img_size=(224,224),
        batch_size=32,
        epochs=20,
        learning_rate=1e-4
    )
