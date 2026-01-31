# Model Management Platform

### Source Data Management
Data Source: [Binance data](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data)

Data Storage: SQLite

### Features data management
Supported Features.

## Model Management
### Model Training Config Page
Manage the training configs with proper version control. All the versions cannot be updated. To provide stable trace back for the model versions.

### Model Training
#### Train the models with given config and code version.
Model config control. Different model relies on different training config. 

#### Model version control

Refine the model storage.
index is an auto-increasing id. To repsent the times of training in that specific day.
`models/${yyyy-MM-dd-index}/${symbol}/`

Use database to control which model to be used in PROD
TBD: design the table for the version
If no model records in DB, then by default use the latest one.

#### Model publish control
1. model need to be trained
2. model need to be tested by forward testing. After certain times with qualified accuracy, promote model to PROD.
3. model accuracy need to be monitored, found x times wrong decision, sunset the model.

### Model forward testing
LLM tagging -> tag the real data to be golden truth.
Human tagging -> human decision to overwrite LLM tagging.
Model predict -> the model decision needs to be tested.

#### Regular forward testing
1. once model was trained. Trigger forward testing regularlly.
    - based on model itself. trigger the forward testing in every 5m, 15m or 1h.
    - If we want to have more aggresive testing, trigger forward testing every mins. 

#### Time Window Flexible

### Model Exception Alert
Mail alert if PROD model got exception during forward testing.