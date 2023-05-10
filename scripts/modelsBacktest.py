
from botsFactoryLib import backtestModels


models = [3,6,12,18,24,36,48,96,168,336]

exposureConstants = [
    5,
    3.75,
    2.5,
    1.25,
    0,
    -1.25,
    -2.5,
    -3.75,
    -5,
]
exposureMultipliers = [
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    500,
    1000,
]

for model in models:
    backtestModels(model, exposureConstants, exposureMultipliers, "BTCUSDT", "1h")

