import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

def train_vision_model():
    print("👁️ Starting CNN Vision Model Training...")
    
    data_dir = r"C:\Users\sidda\Desktop\cattle milk yeild prediction\Clinical_Mastitis_cows_version2\Clinical_Mastitis_cows_version2"
    
    # We expect 'Normal' and 'Abnormal' folders here
    normal_dir = os.path.join(data_dir, "Normal")
    abnormal_dir = os.path.join(data_dir, "Abnormal")
    
    if not os.path.exists(normal_dir) or not os.path.exists(abnormal_dir):
        raise FileNotFoundError(f"Image directories not found in {data_dir}")
        
    print("📸 Loading Image Dataset...")
    
    # Load dataset using tf.keras utility
    batch_size = 32
    img_size = (224, 224)
    
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'  # Normal=0, Abnormal=1
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    class_names = train_ds.class_names
    print(f"✅ Classes detected: {class_names}")

    # Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Base Model (Transfer Learning from MobileNetV2)
    print("🧠 Building Convolutional Neural Network (MobileNetV2)...")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base layers
    
    # Add Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
                  
    print("⚙️ Training CNN for 10 epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\n🏆 Final Vision Model Accuracy: {val_acc * 100:.2f}%")
    
    # Export Model
    models_dir = r"C:\Users\sidda\Desktop\cattle milk yeild prediction\models"
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "milk_vision_model.keras")
    
    model.save(save_path)
    print(f"💾 Success! CNN Vision Model saved securely to: {save_path}")

if __name__ == "__main__":
    train_vision_model()
