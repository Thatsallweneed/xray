## Sameer Shaik's Contribution

### DenseNet121 Implementation
As part of our deep learning approach, I implemented **DenseNet121**, a widely-used convolutional neural network (CNN) architecture, for classifying thoracic diseases using the NIH Chest X-ray Dataset. DenseNet121 is particularly effective for medical image classification tasks as it alleviates the vanishing gradient problem by allowing each layer to receive inputs directly from all previous layers. This architecture helps in learning robust features from the X-ray images, thereby improving classification accuracy.

### Dataset Overview
The **NIH Chest X-ray Dataset** contains **112,120** frontal-view X-ray images from **30,805** unique patients, with images annotated for **14 common thoracic diseases**, including conditions like effusion, infiltration, and atelectasis. The dataset is publicly available and contains metadata for each image, such as disease labels, patient information, and follow-up sequences. Bounding box annotations are also provided for a subset of the images, aiding localization studies.

- **Data Source**: NIH Press Release
- **Dataset Link**: [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### Model Performance
After preprocessing and training the DenseNet121 model, the following results were obtained using the **Receiver Operating Characteristic (ROC) Curve** for multiple disease categories. The area under the curve (AUC) metric is used to evaluate the model's ability to distinguish between the classes. The higher the AUC, the better the model performs in separating the positive class from the negative class.

The ROC curve below shows the performance of the DenseNet121 model across six thoracic disease categories, including effusion, infiltration, mass, nodule, atelectasis, and pneumothorax.

- **Effusion (AUC = 0.80)**
- **Infiltration (AUC = 0.74)**
- **Mass (AUC = 0.71)**
- **Nodule (AUC = 0.70)**
- **Atelectasis (AUC = 0.75)**
- **Pneumothorax (AUC = 0.77)**

These results demonstrate that the DenseNet121 model provides high classification accuracy for the majority of thoracic disease categories. The AUC values indicate that the model is particularly effective in identifying effusion and pneumothorax, with room for further optimization in other categories such as mass and nodule.

### Conclusion
The DenseNet121 model serves as a reliable method for classifying thoracic diseases in X-ray images. However, there is potential for improvement by further fine-tuning the model, addressing class imbalance, and incorporating additional augmentation techniques. Future work could focus on enhancing the classification of underperforming categories and reducing model inference time.
