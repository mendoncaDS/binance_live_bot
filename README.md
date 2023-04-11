
# Bots Factory 🤖
A trading bot class that connects to Binance and tries to make money by predicting a moving average.

## Get Started 🚀  

- Clone the repository:
~~~
  git clone https://github.com/mendoncaDS/binance_live_bot
~~~

- Set up your .env with API key and secret. Connect your github account to https://testnet.binance.vision/ to generate the API credentials
~~~
API_Key_Testnet = {your api key}
Secret_Key_Testnet = {your api secret}

modelName = placeHolderModel
~~~

- Create a conda virtual environment from .yml file:
~~~
  conda env create --file environment.yml
~~~

- Activate the virtual environment

~~~
conda activate liveBot
~~~

- Generate the model, scalers and parameters by running "trainModel.py"
~~~
python scripts/trainModel.py
~~~

- Run the bot
~~~
python scripts/liveBotClass.py
~~~

## How it works 🤓


- "trainModelNB.ipynb" exemplifies the workflow used to develop a model. "trainModel.py" does the same thing but in script format instead of jupyter notebook.

- You can mess around with the functions to make your own.

- Once you are done, make sure the functions in the Jupyter Notebook are available in "botsFactoryLib.py".

- The bot class will use these functions to run the model live, so they should match the custom model you have developed.

- It is recommended that you reserve one repository for every new model.
