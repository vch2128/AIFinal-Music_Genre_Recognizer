# AIFinal-Music_Genre_Recognizer

## Overview
This task is about music genre classification with LSTM. Here are the steps to implement it:

- The data we used save in `music` folder are new data source we generated.
- Convert the music into `.au` files and split each of them into 30-second segments.
- Use `DataProcessing.py` to extract features, turning the audio samples into model-processing data.
- Construct LSTM model and train it by `LSTM.py`
- Process test data and predict their genre in `test.py`

## Prerequisites

- Coding Environment
    - OS: macOS Ventura 13.4
    - Python 3.11.9
- Required packages

Run the following command to install the required packages:
  
  ```
  pip install -r requirements.txt
  ```
 
## Usage
- Download and install packages
    ```
    git clone https://github.com/vch2128/AIFinal-Music_Genre_Recognizer
    cd AIFinal-Music_Genre_Recognizer
    pip install -r requirements.txt
    ```
- Create your test folder with audio samples (.au or .mp3 or .wav)
- Run the model
  
    ```
    python test.py
    ```

    - Expected output:

      ```
        Model available. Loading model...
        Model loading complete.
        Please input the folder you want to organize:
      ```
- Input your test folder name

## Hyperparameters

    ```
    batch_size = 15
    input_dim = 33     # the calculated features
    hidden_dim = 128     # capture hidden features
    layer_num = 1
    output_dim = 9    # 9 genres
    dropout = 0.2
    epochs = 200
    learning_rate = 0.001
    ```

## Experiment results
- Best model
- 

