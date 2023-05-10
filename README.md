
# Bots Factory 🤖
A trading bot class that connects to Binance and tries to make money by predicting a moving average.

You can read about this bot in my medium:

- [Trading Bot in Python](https://medium.com/@mendoncaDS/trading-bot-in-python-99ce77077372)

- [Trading Bot UI in Python](https://medium.com/@mendoncaDS/trading-bot-ui-in-python-571f6710ac5e)

- [Trading Bot Model Selection in Python](https://medium.com/@mendoncaDS/trading-bot-model-selection-in-python-7f01f3769f56)

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

- Generate the model, scalers and parameters by running "trainModel.py" (Note that it uses GPU by default and will probably throw an error if not available. Without GPU it takes too long.)
~~~
python scripts/trainModel.py
~~~

- Run the bot
~~~
run.bat
~~~

## How it works 🤓


- **trainModelNB.ipynb** exemplifies the workflow used to develop a model. **trainModel.py** does the same thing but in script format instead of jupyter notebook.

- You can mess around with the functions to make your own.

- Once you are done, make sure the functions in the Jupyter Notebook are available in **botsFactoryLib.py**.

- The bot class will use these functions to run the model live, so they should match the custom model you have developed.

- Run **modelsBacktest.py** to perform grid search on exposure parameters for live trading. These parameters are set when instantiating a tradingBot class (defined in **liveBotClass.py**)

- These scripts were developed for my medium articles so you probably'll want to tweak the code

- Finally, **trainModelNB.ipynb** is a kind of playground to backtest your models

- Have fun, cheers! 🥳
