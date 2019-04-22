## Agatha Christie Language Model: 
#### Project description: 
The Project contians the scripts needed to create and train a LSTM model on a Text corpa composites of Agath Chrisite novels.
I provided a small processed corpa I downloaded and curated from gutenberg project in the data directory it can be used to investigate the model performance with different hyperparameters like the conditional string length, batch size, etc. 
The training script be runed from a bash shell as shown below.

##### Example1: Create and Train a model on the Demoe dataset: 
$ python Scripts/Raw_data_processing_and_model_training.py --embedding_dimension=128 --num_of_lstm_units=256 --recurrent_dropout=0.2
--input_dropout=0.1 --cond_str_len=120 --batch_size=64 --output=../Model/model_trial1.hf5 --epochs=25

The above code command will create a model with these hyperparameters, train it for 25 epochs and finally save it in models directory. 

##### Example2: Create an inference model: 
Usually, the model is trained using batchs of X size to make use of GPUs, for example, in the above example it is 64 examples, however, during inferences we are interested in making predictions using one conditional string, in most cases. So, a nice trick to address this is to create a new model which accepts batches of size one and then trasfer the weight of the trained model to the newly created model and then use this model for inferences. The following code does exactly that: 

$ python Scripts/Create_inference_model.py --embedding_dimention=128 --num_vocab=69 --num_of_lstm_units=256 --
--input=../Models/AC_languageModel.hf5 output=../Models/inference_model.hf5

##### Example3: Generating Text: 

To generate text from an inference model the following use command: 
python Generate_Text.py --model=../Models/inference_model.hf5 --output=Samples/sample1.txt --len=1000 --temp=0.5 --cond_string="I Love Machine Learning"
