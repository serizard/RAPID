# RAPID: Real-time Aphasia Pattern Interpretation and Diagnosis

![image](https://github.com/user-attachments/assets/9547ee1f-0156-4680-a25f-8407431b4de0)

RAPID is a state-of-the-art tool designed to assist in the diagnosis of aphasia using multimodal inputs such as speech, gesture, and audio data. By leveraging advanced machine learning models, it enables real-time self-diagnosis and generates a comprehensive diagnostic report.

<br/>

## Features

* **Multimodal Analysis**: Integrates speech, gesture, and audio data for precise aphasia type detection
* **User-Friendly**: Designed for easy use at home, reducing the need for frequent hospital visits
* **Interpretability**: Highlights model attention areas, offering transparency in predictions
* **Comprehensive Reports**: Provides detailed diagnostic insights, risk assessments, and personalized recommendations

<br/>

## Installation

To set up the RAPID environment, ensure you have Python 3.8 installed, and then follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/serizard/RAPID.git
   cd RAPID
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
<br/>

## Dataset Building

RAPID utilizes the AphasiaBank Dataset labeled with Western Aphasia Battery (WAB) classifications:

* Control (Normal)
* Fluent Aphasia
* Non-Comprehensive Aphasia
* Non-Fluent Aphasia

Before downloading the dataset, you should get access to [TalkBank](https://talkbank.org/).

Visit the website and check qualification.

<br/>

### Preprocessing Steps

* **Transcription**: Processed using whisper-timestamped
* **Audio Features**: Extracted using opensmile
* **Gesture Analysis**: Conducted using MediaPipe
* **Chunking**: Data is split into chunks of 40 tokens for model input

<br/>

## Training the Model

To train the RAPID model:

1. Navigate to the project directory
2. Run the training script (or you can use .sh scripts):
   ```bash
   python main.py
   ```
   
<br/>

## Model Inference Demo

RAPID offers a simple demo for testing and diagnosis:

1. **Start Test**: Record a video of yourself speaking a predefined text (e.g., the Cinderella story)
2. **View Results**: move to the demo directory and execute ```demo.py```. Make sure that all of required paths and key are involved in ```config.yaml```.

<br/>

## Application Demo

### Running the Streamlit Application

To launch the RAPID user interface via Streamlit:

```bash
cd streamlit-app
streamlit run app.py
```

Also, you have to install dependencies and turn on the dev mode for React.

```bash
cd components/react_components/frontend
npm install
npm start
```

1. **Start Test**: Record a video of yourself speaking a predefined text (e.g., the Cinderella story)
2. **View Results**: Visualize model attention scores and receive a diagnostic report
*Note that you can get actual results only if the remote api server is running.

<br/>

### Setting Up the API for Remote Inference

To configure the remote server for inference:

1. Set up a FastAPI server:
   ```bash
   cd rapid-api-server
   python main.py
   ```
2. Use the provided API endpoints for remote model inference

<br/>

## Acknowledgments

This project builds upon the foundational work from the EMNLP '23 paper:

*Learning Co-Speech Gesture for Multimodal Aphasia Type Detection*  
Authors: Daeun Lee, Sejung Son, Hyolim Jeon, Seungbae Kim, Jinyoung Han  
EMNLP 2023 Proceedings

<br/>

## Dataset Ethics

We sourced the dataset from AphasiaBank with Institutional Review Board (IRB) approval and strictly follow the data sharing guidelines provided by TalkBank, including the Ground Rules for all TalkBank databases based on American Psychological Association Code of Ethics (Association et al., 2002). Additionally, our project adheres to the five core issues outlined by TalkBank in accordance with the General Data Protection Regulation (GDPR) (Regulation, 2018). These issues include addressing commercial purposes, handling scientific data, obtaining informed consent, ensuring deidentification, and maintaining a code of conduct. Considering ethical concerns, we did not use any photographs and personal information of the participants from the AphasiaBank.
