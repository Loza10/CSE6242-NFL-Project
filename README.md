# Football Player Trajectory Prediction
### Description
This repository contains the code and resources for an interactive dashboard built with **Streamlit** to analyze and compare the performance of **five different machine learning models** designed to predict player trajectory (x, y coordinates) in American football.
The models implemented are: neural networks (CNN), transformers, particle filters, and an integrated neural networks/transformer with particle filters approach. The dashboard includes various static visualizations (e.g., error metrics, cross-validation results) and interactive elements,
including two pre-rendered animations/frame-by-frame visualizations of player movement. For computation reasons, the dashboard runs on a small subset of our large dataset. This project was created by Ali Qadri, Westley Cook, Dianze (Jerry) Liu, Zakk Loveall, Keping (Eric) Le, and Amir Javadi. 

All datasets for model training can be found in the Data folder. All datasets needed to render the dashboard can be found in the data folder under the dashboard subdirectory.

### Installation

Users need standard packages including Python 3.10 or above.

To set up the project and install the necessary dependencies, follow these steps:
1.  **Navigate to the Top Directory:**
    Open your terminal and use the `cd` command to navigate to the top-level directory of this repository (e.g., `'Project'`).

2.  **Create a Virtual Environment (Highly Recommended):**
    While not strictly necessary, it is best practice to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate   # On Windows
    ```
    Note: If the above command does not work, try `python3` instead of `python`.

3.  **Install Dependencies:**
    Run the following command to install all required packages listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    Note: Some machine learning packages are large, and this step may take up to 15 to 20 minutes to complete. This step only needs to be done once.
    If you find that you are unable to install the packages in a timely manner, the requirements.txt file has been helpfully segmented into two parts, machine
    learning packages (which are intensive to install), and visualization packages (which are quick to install). You can simply comment out the machine learning
    packages and skip to the dashboard.

### EXECUTION:

## Running Model Training
The original model code is located in the `model` folder, and the datasets are in the `Data` folder. If you wish to train the models yourself and view
the results. Due to the training complexity and computational demands of the transformer model, pretained transformer weights are saved in the transfer
folder and can be loaded directly.

1.  Ensure you are in the top-level directory.

2.  Run `main.py` with your desired model type.
    ```bash
    python main.py
    ```

To launch the interactive Streamlit dashboard:
1.  Change Directory to the Dashboard Folder:
    ```bash
    cd dashboard
    ```

2.  Running the dashboard:
    Execute the following command, specifying a port (e.g., `8576`):
    ```bash
    streamlit run Model_Performance_And_Errors.py --server.port 8576
    ```
    This command will open the dashboard in your default web browser on a local web server.
    Note: If the specified port is already in use or you encounter difficulties (such as a white screen loading endlessly), we recommend incrementing the port number by 1 (e.g., `--server.port 8577`) and running the command again.

3.  Exiting the Dashboard:
    To stop the web server, return to your terminal and press `Ctrl + C`.


## Debugging

    If you find yourself unable to load the dashboard, it is most likely due to one of three reasons, you either did not install all the packages in requirements.txt, or you attempted to launch the dashboard from the wrong subdirectory, or your specified webserver port is already in use.





