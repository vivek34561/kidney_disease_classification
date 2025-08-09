import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your old model WITHOUT compiling to avoid old loss config error
model = load_model('model/model.h5', compile=False)

# Define corrected loss with supported reduction
corrected_loss = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
)

# Compile model with corrected loss
model.compile(
    optimizer='adam',  # use your optimizer here if different
    loss=corrected_loss,
    metrics=['accuracy']
)

# Save fixed model to a new file
model.save('model/model_fixed.h5')
print("Model fixed and saved as model_fixed.h5")
