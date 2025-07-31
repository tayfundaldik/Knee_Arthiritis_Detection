# Knee Arthiritis Detection with U-Net, VGG16 and Random Forest

## Introduction
	
This project focuses on the detection and classification of knee arthritis using real Magnetic Resonance (MR) images collected from a hospital. The goal is to segment inflamed regions and classify the severity of arthritis using medical imaging and machine learning techniques. This approach can assist doctors in diagnosing arthritis more accurately and efficiently.

## Methodology

In this project, I used real knee arthritis Magnetic Resonance (MR) images obtained from a hospital. The dataset consists of two types of images: MR images and corresponding mask images annotated by doctors.

I used U-Net to segment inflamed regions, VGG16 for feature extraction, and Random Forest for final classification.

For implementation, I used libraries such as TensorFlow, Pickle, OpenCV, Matplotlib, Albumentations, and others. I applied k-fold cross-validation during training and plotted the Dice scores and validation losses for each fold. All fold results and segmented data were saved as .npy files.

After segmentation, VGG16 performed feature extraction on the preprocessed data. These extracted features were then used by the Random Forest classifier to determine whether the subject was healthy or affected by arthritis. I applied data augmentation for both U-Net and VGG16 to achieve better results.



### Labeling Images

The first step in the process was labeling the images as either patient or healthy. My approach involved parsing the image filenames to extract relevant metadata. Each filename was structured to include a subject ID, the leg side (e.g., left or right), a health status label (healthy or patient), and, in the case of patients, an image number corresponding to different slices or angles. This structured naming convention allowed me to automatically categorize the images and prepare the dataset for training. And in my project i did the analyze of image names to find which subject is patient or healthy:

```python
def collect_data_pairs(self):
        data_pairs = []

        folders = ['HASTA_SAG', 'HASTA_SOL', 'KONTROL_SAG', 'KONTROL_SOL']

        for folder in folders:
            image_path = os.path.join(self.base_path, folder, 'usg')
            mask_path = os.path.join(self.base_path, folder, 'ROI')

            if os.path.exists(image_path) and os.path.exists(mask_path):
                image_files = glob(os.path.join(image_path, '*.png'))

                for img_file in image_files:
                    img_name = os.path.basename(img_file)
                    mask_name = img_name.replace('image_', 'mask_')
                    mask_file = os.path.join(mask_path, mask_name)

                    if os.path.exists(mask_file):
                        parts = img_name.replace('image_', '').replace('.png', '').split('_')
                        patient_id = parts[0]
                        side = parts[1]
                        condition = parts[2]
                        photo_num = parts[3]

                        data_pairs.append({
                            'image_path': img_file,
                            'mask_path': mask_file,
                            'patient_id': patient_id,
                            'side': side,
                            'condition': condition,
                            'photo_num': photo_num,
                            'category': folder
                        })

        return data_pairs

    def split_data_by_patient(self, data_pairs, test_size=0.2):
        patient_ids = list(set([pair['patient_id'] for pair in data_pairs]))

        train_val_patients, test_patients = train_test_split(
            patient_ids, test_size=test_size, random_state=42
        )

        train_val_pairs = [pair for pair in data_pairs if pair['patient_id'] in train_val_patients]
        test_pairs = [pair for pair in data_pairs if pair['patient_id'] in test_patients]

        return train_val_pairs, test_pairs
```


### Augmentation

Image augmentation is a technique used to increase the size and variability of the training dataset by generating modified versions of existing images. This can involve various transformations such as rotations, shears, brightness adjustments, adding noise, and more.
In this project, I applied brightness adjustments, rotations, flipping, blurring, and grid distortion. Initially, I used only Gaussian noise; however, since Gaussian noise follows a linear and normal distribution, it may not introduce enough variability. To make the learning task more challenging for the model and encourage better generalization, I introduced more diverse augmentation techniques.

**Augmentation in my project:**
```python
def __init__(self, data_pairs=None, data_path=None, batch_size=8, shuffle=True, augment=False, image_size=(256, 256)):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.image_size = image_size

        if data_pairs is not None:
            self.data_pairs = data_pairs
        elif data_path is not None:
            self.image_files = glob(os.path.join(data_path, 'images', '*.npy'))
            self.mask_files = [f.replace('images', 'masks') for f in self.image_files]
            self.data_pairs = None
        else:
            raise ValueError("Either data_pairs or data_path must be provided")

        if self.data_pairs is not None:
            self.indexes = np.arange(len(self.data_pairs))
        else:
            self.indexes = np.arange(len(self.image_files))

        #Augmentation Pipeline#
        if augment:
            self.transform = A.Compose([
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(p=0.3),
            ])
        else:
            self.transform = None

        if shuffle:
            np.random.shuffle(self.indexes)
```

### K-Fold Cross Validation

When building a model, one of common mistakes is overfitting. Overfitting means model learns parameters of a prediction function and testing it on the same data; that would just repeat the labels of the samples that it has just seen would have perfect score but would fail to predict anything useful. To avoid it, it is a common practice that split data as train and test sets. If dataset is small and has imbalanced classes train test split isn't enough. That is where k-fold cross validation is coming out. 

  <img width="1555" height="1036" alt="image" src="https://github.com/user-attachments/assets/0015d4f4-48da-4970-8cc5-c3c330d0de91" /></br>

- **What is K-Fold Cross Validation?**

The K-fold Cross-Validation Method splits the data multiple times (k times) instead of just once into train and test sets. It is the most commonly used model validation method. First, the dataset is divided into k parts. Each time, one part of the dataset is used as the test set, while the remaining k-1 parts are used as the training set. A model is built for each train and test combination of the dataset, and error values are calculated. The average of the error values calculated for each combination is our validation error value. If we test the dataset with a single train and test combination, we do not fully represent the error value of the dataset.



```python
    print("\nStarting K-Fold Cross Validation Training")
    kfold_trainer = SegmentFoldTrain(
        train_val_pairs=train_val_pairs,
        test_pairs=test_pairs,
        k_folds=5,
        model_save_path="models_kfolds1"
    )
```


### U-Net

Before introducing the U-Net Architecture,  it's important to understand the concept of image segmentation. Image segmentation involves dividing an image into meaningful parts to identify specific structures or objects. In this project, it is used to separate inflamed regions from healthy knee tissue in MR images.


<img width="1555" height="1036" alt="u-net-architecture" src="https://github.com/user-attachments/assets/e495f833-deb2-421c-b0d2-112e39510951" />

The architecture has three key parts:

#### Encoder

The encoder captures the input image.

- Uses 3x3 convolutional layers with ReLU activation for scanning the image and find features.
- Uses max pooling for shrink the image size while keeping important information.
- Increasing number of filters at each layer (64 -> 128 -> 256 -> 512) to learn richer features.

This arrangement is repeated several times.

**Encoder part in my project:**

```python
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
```

#### Bottleneck

Bottleneck is the central part of the U-Net architecture connects encoder and decoder blocks. It processes the most compressed representation of the input.

- Receives te smallest feature map with the highest number of filters (1024 in this project).

- Applies two 3x3 convolutions with ReLU and dropout to refine deep features. 

- It does not perform upsampling. That occurs in the decoder.

**Bottleneck part in my project:**

```python
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
```


#### Decoder

Decoder is the counterpart to the encoder in the U-Net architecture. While the encoder compresses the input image into a compact representation, the decoder expands this representation back to the original image size to produce the segmentation mask.

- Combines upsampled features with high-resolution feature maps from the encoder via skip connections, this is key to preserving fine-grained details like edges and boundaries.

- Applies 3×3 convolutions after each merge to refine the combined features.

- The number of filters decreases with depth (512 → 256 → 128 → 64), mirroring the encoder.

**Decoder part in my project:**

```python
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
```
#### Final Output Layer

After the last decoder block, a final 1×1 convolution is used to map the feature vectors to the desired number of output classes .

**Final layer in my project:**

```python
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
```

#### Dice Score

To evaluate segmentation performance, this unet model uses the Dice Score (also known as the Sørensen-Dice Coefficient), rather than pixel-wise accuracy.

**Why Dice Score?**

In image segmentation — especially for tasks like tumor detection, organ delineation, or anomaly localization — pixel-wise accuracy can be misleading due to extreme class imbalance.

For example:

Background pixels: 99% of the image
Foreground (target) pixels: 1% of the image
A model could achieve 99% accuracy simply by predicting "background" everywhere — yet it would fail completely at detecting the target.

The Dice Score addresses this by measuring the overlap between the predicted segmentation and the ground truth:

$$
\text{Dice} = \frac{2 \times |P \cap G|}{|P| + |G|} = \frac{2 \times TP}{|P| + |G|}
$$
 
Where:

$P$ = Predicted mask</br>
$G$= Ground truth mask</br>
$TP$ = True positives



### VGG16 Architecture + Random Forest Classifier
After segmenting the inflamed regions using U-Net, I used the VGG16 architecture to extract high-level features from the segmented (and optionally cropped or masked) MR images.

- **What is VGG16?**

VGG16 is a deep convolutional neural network developed by the Visual Geometry Group at the University of Oxford. It consists of:

- 13 convolutional layers (using 3×3 kernels)

- 5 max-pooling layers

- 3 fully connected layers (at the end)

- Uses ReLU activation and softmax (if used for classification)

<img width="900" height="507" alt="vgg" src="https://github.com/user-attachments/assets/8cee2148-bdd7-443f-98aa-b9826951c2d0" />


One of the key advantages of VGG16 is that it replaces large convolution filters (like 11×11 or 5×5) with multiple 3×3 filters stacked on top of each other. This increases non-linearity and depth while reducing the number of parameters.

**VGG16 in This Project:**

- I used a pre-trained VGG16 model (on ImageNet) without the top classification layer (include_top=False).

- I passed the segmented and resized MR images into VGG16.

- The extracted features were flattened or pooled to reduce dimensionality.

- These features were then used as input to a Random Forest classifier.


```python
def build_vgg16_feature_extractor():

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model
feature_extractor = build_vgg16_feature_extractor()
```

- **Why Random Forest?**

Random Forest is a powerful ensemble learning method that works well on high-dimensional data and is less prone to overfitting compared to individual decision trees.

**In this project:**

- VGG16 provided deep, abstract features from MR images.

- Random Forest used these features to classify whether the image represented a healthy or arthritic knee.

- This hybrid approach combined the strengths of deep learning and traditional machine learning.



```python 
def train_classifier(self, X_train, X_test, y_train, y_test, model_suffix='', random_state=42):

        if len(X_train) == 0:
            print("Error: No data for training!")
            return None, 0, None

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        print(f"\nTraining RF classifier...")
        print(f"Train set: {len(X_train)} examples")
        print(f"Test set: {len(X_test)} examples")

        classifier.fit(X_train_scaled, y_train)

        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Healthy (C)', 'Diseased (P)']))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Healthy (C)', 'Diseased (P)'],
                   yticklabels=['Healthy (C)', 'Diseased (P)'])
        plt.title(f'Random Forest Confusion Matrix {model_suffix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

        feature_importance = classifier.feature_importances_
        top_indices = np.argsort(feature_importance)[-20:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(20), feature_importance[top_indices])
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature Index')
        plt.tight_layout()
        plt.show()

        results = {
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, target_names=['Healthy (C)', 'Diseased (P)'], output_dict=True),
            'feature_importance': feature_importance
        }

        return classifier, accuracy, results
```
## Results

Eventually i ran the models and give model images that model didn't see. 

### Unet Results

<img width="1489" height="1189" alt="download" src="https://github.com/user-attachments/assets/f478ec72-798f-4bb3-a8f9-bb8d7393bd8e" />


As shown in the scores, I compared 5 folds based on Dice Score, Validation Loss, and Validation Accuracy:

- Validation Accuracy was not very useful for fold comparison.

- Validation Loss indicated that Fold 3 had the highest loss, while Fold 4 had the lowest.

- Dice Score showed that Fold 4 achieved the highest score, while Fold 3 had the lowest.

**Conclusion:** Fold 4 provides the most reliable segmentation performance.


### VGG16 Results

<img width="430" height="430" alt="download" src="https://github.com/user-attachments/assets/38f3d82a-46c2-419c-afd2-0ab31bf15b11" /> <img width="430" height="430" alt="download" src="https://github.com/user-attachments/assets/34576ca6-6425-48ff-9884-3e29a1e509be" />



To evaluate the impact of data augmentation on model performance, we compared Random Forest classification results using features extracted from VGG16, both with and without augmentation.

The confusion matrices show that augmentation improves the model’s ability to generalize, especially in detecting diseased cases:

- With augmentation, the model correctly classified 30 diseased and 60 healthy samples, misclassifying only 10 of each.

- Without augmentation, the model correctly classified 27 diseased and 61 healthy samples, but misclassified 13 diseased cases — indicating higher false negatives.

This supports the conclusion that data augmentation helps reduce overfitting and improves sensitivity to the minority class (diseased), which is often harder to classify.

### Running the Project Results

<img width="600" height="600" alt="download" src="https://github.com/user-attachments/assets/5ee1b15b-68e0-471d-b7dd-80cd9dc099d7" /> <img width="600" height="600" alt="download" src="https://github.com/user-attachments/assets/12757b14-835d-452c-9ae5-a2c19dcb28c9" />


The final pipeline for knee arthritis detection works as follows:

#### Segmentation

The U-Net model identifies the joint region and generates a binary mask.

#### Mask Application:

The mask is applied to the original MRI to produce the Masked Image (RGB), isolating the joint area for further analysis.

#### Feature Extraction & Classification:

- VGG16 extracts features from the masked region.

- Random Forest Classifier predicts whether the joint is Healthy or Diseased.

#### Result Visualization:

The Class Probabilities bar chart shows prediction confidence (e.g., Healthy 94.25% vs. Diseased 72.24%).

This approach can assist radiologists and medical researchers by:

- Automatically highlighting affected areas.

- Providing a probabilistic diagnosis to support decision-making.

- Reducing manual effort and enabling early detection of arthritis.

## Future Work

- Experiment with other CNN architectures like AlexNet, GoogleNet, or ResNet.

- Optimize Random Forest hyperparameters or try XGBoost / LightGBM for better performance.

- Investigate ensemble methods to surpass the current 82% accuracy, which is insufficient for clinical deployment.






