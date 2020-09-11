## Traffic Sign Recognition TensorFlow

Overview
---
In this project, various types of neural networks are trained aiming to obtain accurate results for traffic signs classification. The DL libraries used are TensorFlow, Keras, and Scikit-Learn. The dataset used in this project is [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

Usage
---

1. Clone the project
2. Download and unzip the [dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) into the root directory of this project
3. Make sure all the necessary libraries and dependencies are installed
```sh
pip3 install --update numpy scikit-image scikit-learn tensorflow
```
4. Run script
```sh
python3 dnn.py -d dataset_dir_path [-m path_to_save_model] [-w path_to_save_weights_bias]
```
