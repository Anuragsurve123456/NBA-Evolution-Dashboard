# 🏀 Visualizing the Evolution of the NBA (1946–2023)

This project is an interactive Python-based data visualization dashboard that explores the long-term evolution of the National Basketball Association (NBA).
It was developed as the Final Term Project (FTP) for DATS 6401 – Visualization of Complex Data at The George Washington University.

The dashboard combines statistical analysis, exploratory visualization, storytelling, and interactivity to reveal how NBA strategies, scoring patterns, and team dominance have evolved across eras.

---

## 📌 Project Objectives

* Apply Python visualization techniques learned in the course to a real-world dataset
* Perform data cleaning, statistical validation, and exploratory analysis
* Build an interactive web-based dashboard using Dash
* Allow users to update plots dynamically without re-running code
* Deploy the dashboard in a production-ready environment using Google Cloud Platform

---

## 📊 Dataset Description

The dataset consists of NBA game-level data from 1946 to 2025, where each row represents a single game.

Key features include:

* Scoring metrics such as points, field goals, and free throws
* Playmaking metrics such as assists and rebounds
* Shooting efficiency metrics including FG%, 3P%, and FT%
* Shot selection metrics such as three-point attempt share
* Categorical variables including teams, season type, and eras

This dataset is well-suited for complex visualization due to its size, temporal depth, and mix of numeric and categorical variables.

---

## ⚙️ Installation & Setup

1. Clone the repository
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd nba-dash-dashboard

2. Create a virtual environment (recommended)
   python -m venv venv

Activate the environment:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

3. Install required packages
   pip install -r requirements.txt

---

## 📂 Dataset Setup

Place the dataset file at:
nba-dash-dashboard/data/game.csv

If the dataset is stored elsewhere, you can set an environment variable:
DATA_PATH=/full/path/to/game.csv

The application automatically detects the dataset location.

---

## ▶️ Running the Dashboard

Run the application using:
python app.py

Then open a browser and go to:
[http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 🌐 Live Application Link

The interactive NBA dashboard has been successfully deployed on Google Cloud Platform (Cloud Run) and is publicly accessible at https://dashapp-718572610311.us-east1.run.app/
. This live deployment allows users to explore all dashboard features—including interactive filtering, team-level storytelling, animated bar chart races, and statistical visualizations—directly in the browser without any local setup. The hosted application demonstrates a complete end-to-end pipeline from data preprocessing and visualization to a production-ready, cloud-deployed analytics system.

---

## 🖥️ Dashboard Features

The dashboard is organized into three main sections:

Home

* Dataset overview
* Sample rows
* Summary statistics

Data Cleaning & Analysis

* Missing value inspection
* Outlier detection
* Data transformation using Z-scores
* Correlation analysis
* Principal Component Analysis (PCA)
* Normality testing
* Numeric and categorical visualization explorers

NBA Evolution

* League-wide trend analysis
* Three-point revolution analysis
* Home vs away scoring comparison
* Extreme scoring outliers
* Animated bar chart race of team dominance
* Team-level storytelling with auto-generated insights
* Downloadable figures and datasets

---

## ☁️ Deployment

The dashboard was containerized using Docker and deployed using Google Cloud Platform (Cloud Run), making it accessible as a production-ready web application.

---

## 📚 Technologies Used

* Python
* Dash and Dash Bootstrap Components
* Plotly
* Pandas and NumPy
* Scikit-learn
* SciPy
* Google Cloud Platform

---

## 📄 Academic Note

This project was developed individually for academic purposes.
All code was written manually, and all visualizations were generated entirely in Python in accordance with course guidelines.

---

## 👤 Author

Anurag Surve
M.S. in Data Science
The George Washington University

---

## 📌 Acknowledgments

Course Instructor: Dr. Reza Jafari
Course: DATS 6401 – Visualization of Complex Data
