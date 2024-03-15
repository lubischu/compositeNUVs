# compositeNUVs
This repository contains code to perform various model-based signal processing tasks. It is mainly built around two novel composite NUV priors, providing the necessary functionalities for a powerful model selector mechanism and a covariance estimation method. All the provided libraries and notebooks were developed during a master's project at ETH Zurich, the final report of which can be found [here](https://example.com). Note that all the simulations described in this report can be reproduced by [this](https://github.com/lubischu/compositeNUVs/blob/main/notebooks/simulationsInReport.ipynb) notebook.

## Installation
1. Clone git repository to local machine.
   
   ```bash
   git clone https://github.com/lubischu/compositeNUVs.git
   ```

2. Create and activate a virtual environment with specified requirements.

   For Unix/Linux/macOS:
   
   ```bash
   python -m venv cnenv
   source cnenv/bin/activate
   pip install -e .
   ```

   For Windows:
   
   ```bash
   python -m venv cnenv
   cnenv\Scripts\activate
   pip install -e .
   ```
