# Christmas Songs Dashboard 🎄

**Description**:
This project is an interactive web application built with Dash to explore Christmas songs on the Billboard Hot 100. The dashboard enables users to analyze song trends, performances, and comparisons over time using dynamic visualizations and data tables.

### Features:
1. **Data Exploration**:
   - Load and clean Billboard data for Christmas songs.
   - Perform descriptive statistical analysis on songs and performers.

2. **Visualizations**:
   - Line charts for yearly song trends.
   - Scatter plot of peak positions over time.
   - Bar chart showcasing top performers with the most songs.

3. **Interactive Elements**:
   - Searchable and sortable data table for top songs.
   - Artist comparison with dropdown selection.
   - Seasonal performance and timeline visualizations.

4. **Key Metrics**:
   - Most successful and enduring songs.
   - Consistent performers with at least five appearances.

### How to Run:
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the app using `python app.py`.
4. Open the app in a browser at `http://127.0.0.1:8050`.

### Dataset:
The app uses a cleaned dataset (`hot100_clean.csv`) derived from the Billboard Hot 100 and a previous list of Holiday songs features on the Billboard Hot 100 from 1958 - 2024 derived from kaggle, https://www.kaggle.com/datasets/sharkbait1223/billboard-top-100-christmas-carol-dataset.  
Data preparation includes renaming columns, handling missing values, and calculating new features like `year`, `decade`, and `is_december', filtering the Hot 100 data on the kaggle dataset and other Holiday keywords.

