# Uno_card_color_shape_detection
 
##Uno_Card_Detection

This model aims to detect which card has been thrown into the camera.

##Dataset :

It has 15 classes, each class with 500 images fed to the model

The dataset is made from videos recorded for a specific class, and a code attached with this folder name 'extract frames from videos' helped to create images for each class

Classes:"Card_0","Card_1","Card_2","Card_3","Card_4","Card_5","Card_6","Card_7","Card_8","Card_9","Draw_2+","Draw_4+","Reverse","Skip_Card","Wild_Card"

##Implementing the training script with Keras : 

Using different learning rates, comparing the accuracy and loss rates to find optimal learning rates, and fine-tuning the VGG16 CNN on the built dataset.

##Import necessary packages including :

matplotlib: For plotting (using the "Agg" backend to save plot images to disk).

tensorflow: Imports including our VGG16 CNN, data augmentation, layer types, and SGD optimizer.

scikit-learn: Imports including a label binarizer, dataset splitting function, and an evaluation reporting tool.

keras callback: callback lead to faster convergence and typically require fewer experiments for hyperparameter updates

cv2: OpenCV for preprocessing and display.

##The next section is :
 
Grabs paths to all images in the dataset. Then initialize two synchronized lists to hold the image (data and labels)
Loop over imagePaths, while:
Extracting the class label from the path.
Loading and preprocessing the image. Images are converted to RGB channel ordering and resized to 224×224 for VGG16.
Adding the preprocessed image to the data list.
Adding the label to the labels list.
Performs a final preprocessing step by converting the data to a "float32" data type NumPy array.
Similarly, it converts labels to an array so that one-hot encoding can be performed
Before setting up the VGG16 framework, construct training, testing, and validation splits.

##Next stage is to setup VGG16 model for fine-tuning :

Load VGG16 using pre-trained ImageNet weights (but without the fully-connected layer head).

Create a new fully-connected layer head, which adds the new FC layer to the body of VGG16.

Only the new FC layer head is used for training so that it fits the classes of the dataset

Then compile the model with the Stochastic Gradient Descent (SGD ) optimizer and try different learning rates to choose the best.

##Implementing our Uno Card detection script : 

Load necessary packages and modules. In particular, use deque from Python’s collections module to assist with our rolling average algorithm.

By performing rolling prediction accuracy, we’ll be able to “smooth out” the predictions and avoid “prediction flickering.”
Duplicate the frame for output purposes and then preprocess it for classification. The preprocessing steps are and must be, the same as those performed for training.

Perform inference and add the predictions to our queue.

Performs a rolling average prediction of the predictions available in the Q.

Then extract the highest probability class label to annotate the frame.

##Output 

The model produces 87%, which is not an overfitting accuracy. A real-time test video is attached to this.

##Color Detection 

Only the card image should be cropped and passed to the code for the color detection logic. For the same, a yolov5 model is trained with only one class

##Yolov5 model

1. A dataset with class – 'Card'
2. Labelled them using the Yolo Label tool
3. All the necessary libraries to install for training yolov5 are commented in the notebook
3. Images annotation should be disabled in the code, otherwise it will lead to misclassification
4. Finally, developed a yolov5-based object detection algorithm

The ROI from this model is the input for color detection

Results : Attached test.mp4 for card detection inference as it has some false predictions we are filtering out results with only 0.50 confidence or above

## Color detectionb Logic

This Python script imports several libraries such as NumPy, OpenCV (cv2), PyTorch, webcolors, and scikit-learn.

It also defines several functions used to process images and extract color information.

The function get_colour_name(requested_colour) takes an RGB color value and returns the closest named color from the CSS3 color list. It does this by first attempting to convert the RGB value to a named color using webcolors.rgb_to_name() function.

If that fails, it calls the closest_colour() function to find the closest color based on Euclidean distance.
The closest_colour() function loops through all the named CSS3 colors and calculates the Euclidean distance between the requested color and each named color. It returns the name of the color that has the smallest distance.

The centroid_histogram(clt) function takes a KMeans clustering object (clt) and calculates a histogram of the number of pixels assigned to each cluster. It returns the normalized histogram.

The plot_colors(hist, centroids) function takes a histogram and a list of cluster centroids and creates a bar chart showing the relative frequency of each color in the image. It initializes an empty bar chart and then loops over each cluster, plotting a rectangle for each color in the bar chart.

Overall, these functions can extract color information from images and create visualizations of the color distribution in those images.

As the dominant color is white for all cards, we are taking the second dominant color to predict the card color. 


