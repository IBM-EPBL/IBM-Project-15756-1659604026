# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'Data',  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['Apple___Black_rot','Apple___healthy','Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight',
'Peach___Bacterial_spot','Peach___healthy','Pepper_bell___Bacterial_spot','Pepper_bell___healthy','Potato___Early_blight',
'Potato___healthy','Potato___Late_blight','Tomato___Bacterial_spot','Tomato___Late_blight',
'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')