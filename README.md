# DeepHumanDetector

## Information
This is a project to develop a deep learning model that detects people in images with high accuracy in real-time. It uses a large dataset of people to predict two classes: `head` and `person`. The following is a demo video showing the actual detection of `head` and `person` from a video.Both videos use the same deep learning model for inference.

https://github.com/tomohaya23/DeepHumanDetector/assets/52773164/20d2a48c-3df0-4a89-8d3a-d04005b50e93

https://github.com/tomohaya23/DeepHumanDetector/assets/52773164/cfef3ff2-4d3c-4648-9d29-35e1005e023f

## 1. Dataset
I am using two datasets for training: the MSCOCO dataset (https://cocodataset.org/) and my own proprietary dataset. It is important to note that I have extended the MSCOCO dataset by adding my own data and re-labeling all the images to fit my specific needs. I do not intend to make these two datasets publicly available.
  
  - The total number of training data and instances are as follows:
    ```
    - Training data
      - Total: Approximately 320,000 images
      - Total: Approximately 1.7 million instances
    ```

## 2. Requirements
The development environment for this project uses Ubuntu 22.04.4 LTS. The Python version is Python 3.10.12, and the CUDA version is 12.1. To install the same Python packages as this project, run the following command:

<details><summary>Installation</summary>

```
pip install torch torchvision torchaudio
pip install tqdm
pip install pyyaml
pip install simplejson
pip install onnx
pip install opencv-python
pip install onnxruntime
```

</details>

## 3. Data preparation
### Create your own dataset
**Step 1** 
Prepare your own dataset with images and labels first.

**Step 2**
Create a directory for your prepared custom dataset under the "dataset" directory.
e.g. dataset/your_dataset

**Step 3** 
Create a symbolic link to the directory containing the dataset's image data under the custom dataset directory you created, as follows:

```
ln -s /path/to/your_dataset/images ./dataset/your_dataset/images
```

Alternatively, move the dataset images directly under the "dataset" directory.

**Step 4**
Create a file named "load_dataset.py" under the directory you created in Step 2.
e.g. dataset/your_dataset/load_dataset.py

After completing the steps up to this point, the directory structure under the 'dataset' directory will be as follows:

```
DeepHumanDetector
└── dataset/
    └── your_dataset/
        ├── images
        └── load_dataset.py
```

**Step 5**
Add the code to read the image file paths and annotation information of your prepared custom dataset in "load_dataset.py". Refer to the contents of "dataset/your_dataset/load_dataset.py" for guidance.

**Step 6**
Update "training/load_datasets.py" to call the "dataset/your_dataset/load_dataset.py" file you created. Add the following code:
```
# Add your own dataset here
# For example:
if dataset_name == "your_dataset":
    from dataset.your_dataset.load_dataset import load_data

    train_data_your, train_label_your = load_data(os.path.join("dataset", dataset_name))
    train_data.extend(train_data_your)
    train_label.extend(train_label_your)
```

## 4. Training
For model training, there are two approaches: training from scratch using your prepared custom dataset, or fine-tuning using the pre-trained weights that I have provided. Below, I will list the commands for each case. Please download the pre-trained weights from the following link and use them.
https://github.com/tomohaya23/DeepHumanDetector/releases/download/v.0.1.0/best_model.pth

  - Training from scratch without using pre-trained weights
    ```
    python train.py --datasets your_dataset
    ```

  - Fine-tuning using pre-trained weights
    ```
    python train.py --datasets your_dataset --latest_checkpoint best_model.pth
    ```

## 5. Export
Export the weights of the model trained with PyTorch to ONNX format. Follow the command below to perform the conversion.

```
python torch2onnx.py --model_path best_model.pth
```

After executing the command, an "onnx" directory will be created, and a "best_model.onnx" file will be generated within that directory. In my use case, I assume that the ONNX model will be used on a CPU, but if you want to use it on a GPU, please modify "torch2onnx.py" as needed.

## 6. Inference
Execute the command as follows:
  - For movies:
    ```
    python detect.py --input_type movie --movie_path /path/to/your/movie.mp4 --model_path onnx/best_model.onnx
    ```
  - For images:
    ```
    python detect.py --input_type image --image_list_file /path/to/your/image_list.txt --model_path onnx/best_model.onnx
    ```

    For "--image_list_file", please specify a text file containing the file paths of the images. You can refer to "image_list.txt" as an example.

## Reference
  - https://cocodataset.org/
  - https://github.com/Megvii-BaseDetection/YOLOX
