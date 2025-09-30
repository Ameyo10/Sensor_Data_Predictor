# Human Activity Recognition with Sensor Data + LLM Integration

This project integrates a CNN+RNN deep learning model for sensor-based human activity recognition with a GPT-2 language model to enable natural language interaction. Users can query the system in plain English (e.g., “Predict the activity from this sensor input” or “Show me example data for Walking”), and the assistant provides predictions or visualizations accordingly.

# Project Structure

Sensor_prediction.ipynb

Trains a CNN+RNN hybrid model on sensor data (e.g., accelerometer, gyroscope).

Tasks:

Data preprocessing & train/test split

Model architecture: CNN (feature extraction) + RNN (temporal patterns)

Training, evaluation, and saving the model for later use

LLM_for_Sensor_prediction.ipynb

Fine-tunes a GPT-2 model for handling user queries in natural language.

Tasks:

Load pretrained GPT-2

Fine-tune on project-specific prompts/responses

Save tokenizer and model weights

Integrating_models.ipynb

Combines the trained CNN+RNN model and GPT-2 model.

Provides a query interface where users can:

Ask the model to predict activities from raw sensor arrays

Request example arrays for activities (e.g., Walking, Sitting, Standing)

Interact via plain English commands

# How It Works

Sensor data modeling:
The CNN+RNN model learns temporal dependencies in sensor data to classify activities like:

Walking

Sitting

Standing

(Other dataset-specific activities)

Language interface:
The GPT-2 model maps natural language queries to backend functions.

Example: “Predict the activity” → system forwards data to CNN+RNN → returns predicted class.

Example: “Show me data for Walking” → system fetches example array and visualizes it.

Integration:
The assistant class in Integrating_models.ipynb coordinates between sensor model outputs and GPT-2 text responses.

# Setup

Clone the repo and install dependencies:

git clone <your_repo_url>
cd <repo>
pip install -r requirements.txt


Train CNN+RNN model:
Open Sensor_prediction.ipynb → run all cells → saves cnn_rnn_model.pth.

Fine-tune GPT-2:
Open LLM_for_Sensor_prediction.ipynb → run all cells → saves gpt2_model.pth and tokenizer.

Integrate & test:
Open Integrating_models.ipynb → interact with assistant using queries.

# Example Queries
assistant.query_llm("Predict the activity", sensor_data=x_test.iloc[0].values)

assistant.query_llm("Show me data for Walking")

assistant.query_llm("What does the array look like for Sitting?")

# Future Work

Expand dataset with more activities

Use larger LLMs for richer query understanding

Deploy as a REST API or streamlit app for interactive demos
