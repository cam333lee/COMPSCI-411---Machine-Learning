#Cameron Lee
#5/4/2023
#Kaggle Sea Animals Dataset

from tensorflow.keras import preprocessing
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
import numpy as np

def cnn1():

    #Training Set
    training_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
##    for e in training_set:
##        print(e)
##        a = input() #to pause

##    for e in training_set:
##        print(e[1][0])
##        plt.imshow(e[0][0]/255)
##        plt.show()

    #Test Set
    test_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                          validation_split=0.2,
                                                          subset="training",
                                                          label_mode="categorical",
                                                          seed=0,
                                                          image_size=(100,100))
    print("Classes:", training_set.class_names)

    #Now can build CNN
    m1 = Sequential()
    m1.add(Rescaling(1/255)) #Rescaling Layer
    m1.add(Conv2D(32, kernel_size=(3,3),
                 activation = 'relu',
                 input_shape=(100,100,3)))
    m1.add(MaxPooling2D(pool_size=(2,2)))
    m1.add(Dropout(0.2))
    #Add next convolutional layer
    m1.add(Conv2D(64, (3,3),activation='relu'))
    m1.add(MaxPooling2D(pool_size=(2,2)))
    m1.add(Dropout(0.2))
    m1.add(Conv2D(128, kernel_size=(3,3),activation="relu"))
    m1.add(MaxPooling2D(pool_size=(2,2)))
    m1.add(Dropout(0.2))
    #Now we need to combine the outputs of last layer and then flatten the structure
    #Need to change the nodes of a 2D layer to 1D layer
    m1.add(Flatten())

    #Add traditional nerual network Dense layer
    m1.add(Dense(128, activation="relu"))
    m1.add(Dropout(0.2))

    #Ouput layers
    m1.add(Dense(5, activation="softmax"))
    
    #Done Building
    #Need to compile to specify settings
    m1.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    #training CNN
    epochs = 35
    print("Training:")
    for i in range(epochs):
        history = m1.fit(training_set, batch_size=32, epochs=1, verbose=0)
        print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

    m1.summary()

    #Test model's accuracy on test data
    print("Testing:")
    score = m1.evaluate(test_set, verbose=0)
    print("Test Accuracy:", score[1])

    #Save the model
    print("Saving the model in my_cnn1.h5")
    m1.save("my_cnn1.h5")

def cnn2():
    #Training Set
    training_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split=0.2,
                                                              subset="training",
                                                              label_mode="categorical",
                                                              seed=0,
                                                              image_size=(100,100))
    #Test Set
    test_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                          validation_split=0.2,
                                                          subset="training",
                                                          label_mode="categorical",
                                                          seed=0,
                                                          image_size=(100,100))
    print("Classes:", training_set.class_names)

    #Now can build CNN
    m2 = Sequential()
    m2.add(Rescaling(1/255))
    #Add first convolutional layer with larger filter size 
    m2.add(Conv2D(32, kernel_size=(4, 4), activation='relu',input_shape=(100,100,3)))
    
    m2.add(MaxPooling2D(pool_size=(2, 2)))
    #Changed dropout layer to 0.15
    m2.add(Dropout(0.15))
    m2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    m2.add(MaxPooling2D(pool_size=(2, 2)))
    #Changed dropout layer to 0.15
    m2.add(Dropout(0.15))
    m2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    m2.add(MaxPooling2D(pool_size=(2, 2)))
    #Changed dropout layer to 0.15
    m2.add(Dropout(0.15))
    m2.add(Flatten())
    m2.add(Dense(128, activation='relu'))
    #Changed dropout layer to 0.15
    m2.add(Dropout(0.15))
    m2.add(Dense(5, activation='softmax'))

    #setting and training
    m2.compile(loss="categorical_crossentropy", metrics=['accuracy'])

    epochs = 30
    print("Training:")
    for i in range(epochs):
        history = m2.fit(training_set, batch_size = 32, epochs = 1, verbose=0)
        print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

    #Testing
    print("Testing:")
    score = m2.evaluate(test_set, verbose=0)
    print("Test Accuracy:", score[1])

    #Save the model
    print("Saving the model in my_cnn2.h5")
    m2.save("my_cnn2.h5")
    

def fine_tune():    
    training_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split = 0.2,
                                                              subset="training",
                                                              label_mode="categorical",
                                                              seed = 0,
                                                              image_size=(100, 100))
    test_set = preprocessing.image_dataset_from_directory("sea_animals",
                                                              validation_split = 0.2,
                                                              subset="validation",
                                                              label_mode="categorical",
                                                              seed = 0,
                                                              image_size=(100, 100))
    print("Classes:", training_set.class_names)

    #Load general pre-trained model
    #WROTE THIS IN PYTHON PROMPT SINCE DOWNLOADING FOR FIRST TIME
    base_model = VGG16(weights="imagenet",include_top=False)

    #Can continue to fine tune in idle now 
    #output la
    x = base_model.output #output layer of base model
    print("Printing x:", x)
    x=GlobalAveragePooling2D()(x) #x reduces dimensions by averaging
    x=Dense(1024, activation='relu')(x) #fully connected layers

    #Note: This is not a Sequential model, this is the Functional API
    #that gives me control to connect layers. "Call" one layer on another layer object.

    output_layer = Dense(5, activation='softmax')(x)
    
    #This is the model we will train to fine tune
    m = Model(inputs=base_model.input, outputs=output_layer)

    #We don't want the model to re-learn the weights of the base_model
    #train only the top layers (which were randomly initialized)
        #AKA freeze all convolutional base model layers
    for layer in base_model.layers:
       layer.trainable = False

    m.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    #Training
    epochs = 12
    print("Training:")
    for i in range(epochs):
        history = m.fit(training_set, batch_size=32, epochs=1, verbose=0)
        print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

    #Testing
    print("Testing:")
    score = m.evaluate(test_set, verbose = 0)
    print("Test Accuracy:", score[1])

    #saving the model
    print("Saving the model in my_fine_tuned.h5")
    m.save("my_fine_tuned.h5")

def test_image(model, image_file):
    #Load Image
    img = preprocessing.image.load_img(image_file,target_size=(100,100))
    img_arr = preprocessing.image.img_to_array(img)

    #See image
    plt.imshow(img_arr.astype("uint8"))
    plt.show()

    #Classify image
    img_cl = img_arr.reshape(1,100,100,3)
    score=model.predict(img_cl)
    print(score.round(3))

    
cnn1()
cnn2()
fine_tune()
#test_image()

print("CNN2 WAS BETTER MODEL")
fine_tuned = load_model("my_fine_tuned.h5")
#my_cnn1 = load_model("my_cnn1.h5")
my_cnn2 = load_model("my_cnn2.h5")

print("Image 1 (Dolphin)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/Dolphin.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/Dolphin.jpg')
print()

print("Image 2 (Dolphin)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/dolphin2.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/dolphin2.jpg')
print()


print("Image 3 (Penguin)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/penguin1.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/penguin1.jpg')
print()


print("Image 4 (Penguin)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/penguin2.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/penguin2.jpg')
print()

print("Image 5 (Sea Ray)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/searay1.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/searay1.jpg')
print()

print("Image 6 (Sea Ray)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/searay2.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/searay2.jpg')
print()

print("Image 7 (Seal)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/seal1.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/seal1.jpg')
print()
 
print("Image 8 (Seal)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/seal2.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/seal2.jpg')
print()                         

print("Image 9 (Shark)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/shark1.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/shark1.jpg')
print()


print("Image 10 (Shark)")
print("CNN 2 Model:")
test_image(my_cnn2, '/Program Files/Python311/test_animals/shark2.jpg')
print("fine_tuned Model:")
test_image(fine_tuned, '/Program Files/Python311/test_animals/shark2.jpg')
print()

    
                                                            
