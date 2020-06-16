# WFaceNet

## Introduction
* Train the model on CASIA-WebFace dataset, and evaluate on LFW dataset.

## Requirements

* Python 3.6.10
* pytorch 0.4.1
* CUDA 9.2
* OpenCV-python
* scipy

## Usage

### Part 1: Preprocessing

* Download the aligned images at [CASIA-WebFace@BaiduDrive](https://pan.baidu.com/s/1rLfuxHdu0prH-_2B0yWaBg)(code:f9xx) and [LFW@BaiduDrive](https://pan.baidu.com/s/1XZyRmzTo8j699Ezpg-VySQ)(code:82gh).

### Part 2: Train

  1. Change the **CASIA_DATA_DIR** and **LFW_DATA_DAR** in `config.py` to your data path.
  
  2. Train the WFaceNet model. 
  
        **Note:** The default settings set the batch size of 128, use 1 gpu and train the model on 70 epochs. You can change the settings in `config.py`.
      ```
      python3 train.py
      ```
      
### Part 3: Test

  1. Test the model on LFW. `train.py` test the model on LFW after each train epoch complete automatically.
    
          

## Results

  * You can just run the `lfw_eval.py` to get the result, the accuracy on LFW like this:
```
    1    99.33
    2    99.33
    3    99.67
    4    98.83
    5    98.83
    6    99.67
    7    98.83
    8    99.50
    9    99.83
   10    99.67
   
   AVE    99.35
```
  ```
      python3 lfw_eval.py --resume --feature_save_dir
      ```
      * `--resume:` path of saved model
      * `--feature_save_dir:` path to save the extracted features (must be .mat file)
  ```

## Reference resources

  * [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
  * [MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)
