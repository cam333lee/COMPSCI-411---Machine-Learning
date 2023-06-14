Task 1: Your own CNN architectures:

Try two different CNN architectures (one can be from the my_cnn function in the posted A4.py program). You can change the layers, change the number or size of filters, with and without dropout, etc. to get different architecture(s).

Divide the data into training and test set as in the A4.py program. Train each model till the training accuracy does not seem to improve over epochs. Test it on the test data. Save each model in .h5 file.

Task 2: Fine-tuning a pre-trained CNN architecture:

Fine-tune VGG16 architecture to classify your five sea animals as explained in the slides (you can use the fine_tune function from posted A4.py program).

Divide the data into training and test set as in the A4.py program (it will be same as in Task 1). Train the model till the training accuracy does not seem to improve over epochs. Test it on the test data. save the model in .h5 file.

Task 3: Error Analysis:

Search on the Internet and download 10 images of the sea animals of your 5 chosen categories (they could be all from one category or from more categories). Test each image using the best model of Task 1 and the model of Task 2 (you can use your saved models and the test_image function from the posted A4.py program). Qualitatively see what images the models get correct and incorrect, and if there is any pattern.
