import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model(input_shape=(128, 128, 3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("Setting up dataset generators...")
    
    BATCH_SIZE = 32
    IMG_SIZE = (128, 128)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2
    )
    
    if os.path.exists('dataset'):
        train_generator = train_datagen.flow_from_directory(
            'dataset/', 
            target_size=IMG_SIZE, 
            batch_size=BATCH_SIZE, 
            class_mode='categorical', 
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            'dataset/', 
            target_size=IMG_SIZE, 
            batch_size=BATCH_SIZE, 
            class_mode='categorical', 
            subset='validation'
        )
        
        # Dynamically get class labels and number of classes from the folder structure
        num_classes = len(train_generator.class_indices)
        print(f"Detected classes: {train_generator.class_indices}")
        
        model = create_model(input_shape=(128, 128, 3), num_classes=num_classes)
        model.summary()
        
        print("Training model across all categories...")
        model.fit(train_generator, validation_data=val_generator, epochs=5)
        
        model.save('model.h5')
        print("Trained multi-class model successfully saved as model.h5!")
    else:
        print("Dataset folder not found!")
