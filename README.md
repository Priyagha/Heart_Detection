# Heart Bounding Box Prediction from X-Ray Images using CNN

In this project, we develop and train a model to predict bounding boxes around the heart in X-ray images. Understanding the position and size of the heart can provide crucial diagnostic information for various cardiac conditions and associated symptoms. For instance, cardiomyopathy, characterized by an enlarged heart, and conditions like pneumothorax or atelectasis, which may lead to shifts in the heart's position, are examples where accurate heart localization can aid in early detection and treatment planning.

## Dataset

To address the challenge of heart localization in X-ray images, we manually annotated the position of the heart in 469 images sourced from the Arizona Pneumonia Detection Challenge dataset. These annotations serve as ground truth labels for training our model. The objective is to develop a model capable of accurately predicting bounding boxes around the heart in all chest X-ray images.

## Preprocessing

Preprocessing plays a crucial role in preparing the data for model training. We resize the images to 224 squared, ensuring compatibility with the model architecture. It's important to rescale the bounding box coordinates accordingly to prevent discrepancies. Additionally, we standardize pixel values to the [0, 1] interval and compute the training mean and standard deviation for dataset normalization.

## Dataset Handling

Unlike classification tasks, where pre-made dataset classes are readily available, we must create a custom dataset class tailored to our bounding box prediction task. This class must load X-ray images along with their corresponding bounding box coordinates based on subject IDs. Furthermore, it enables the application of set normalization, transformations, and augmentation pipelines to ensure consistency in data processing.

Data augmentation is essential to increase the model's robustness and generalization capability. We employ augmentation techniques such as random contrast changes, scaling, rotations, and translations. It's crucial to ensure that both X-ray images and bounding boxes undergo identical augmentations to maintain alignment between them. The Oak Library provides support for performing synchronized augmentations on both image and bounding box data.

## Model Architecture and Training

We utilize the ResNet-18 architecture as the base network, modifying the input channels from 3 to 1 to accommodate grayscale X-ray images. The model is trained to predict four coordinates representing the bounding box: the top-left and bottom-right corners. We employ the mean squared error (L2 loss) as the loss function and the Adam Optimizer with an initial learning rate of 1e-4 for model training. The model is trained for 50 epochs to optimize its performance in predicting accurate bounding boxes around the heart.