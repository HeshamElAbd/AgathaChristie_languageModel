## Agatha Christie Language Model: 
#### Project description: 
The Project contians the scripts needed to create and train a LSTM model on a Text corpa composites of Agath Chrisite novels.
I provided a small processed corpa I downloaded and curated from gutenberg project in the data directory it can be used to investigate the model performance with different hyperparameters like the conditional string length, batch size, etc. 
The training script be runed from a bash shell as shown below.

##### Example1: Create and Train a model on the Demoe dataset: 
$ python Scripts/Raw_data_processing_and_model_training.py --embedding_dimension=128 --num_of_lstm_units=256 --recurrent_dropout=0.2
--input_dropout=0.1 --cond_str_len=120 --batch_size=64 --output=../Model/model_trial1.hf5 --epochs=25

The above code command will create a model with these hyperparameters, train it for 25 epochs and finally save it in models directory. 
