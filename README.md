# AI Data Analysis System

## Developed by
**Name:** Manish Paneru  
**Title:** Data Analyst  
**Passion:** Creating AI Agent programs to be the laziest analyst possible

## Overview
The AI Data Analysis System is a sophisticated data analysis and reporting tool designed to automate the process of analyzing data and generating professional reports. The system can ingest CSV datasets, perform data cleaning, execute comprehensive analysis, create visualizations, and produce beautifully formatted PDF reports—all with minimal human intervention.

This project leverages the CrewAI framework and other modern data analysis libraries to create a workflow that simulates a team of analysts working together to answer business questions.

## Key Features

- **User-friendly Command Line Interface**: Simply input your name, business question, and dataset path
- **Comprehensive Data Analysis**: Performs data profiling, statistical analysis, and correlation identification
- **Advanced Visualizations**: Creates professional-quality data visualizations including:
  - Correlation heatmaps
  - Distribution histograms
  - Bar charts of key metrics
  - Comparative analyses
  - Scatter plots with trend lines
- **Professional PDF Reporting**: Generates beautifully formatted PDF reports with:
  - Cover page and table of contents
  - Executive summary
  - Detailed analysis sections
  - High-quality visualizations with detailed explanations
  - Actionable recommendations
  - Professional styling with color themes
- **Customization Options**: Adjust report appearance with different color themes
- **Author Attribution**: Includes user's name as the report author

## Technologies Used

- **Python**: Core programming language
- **CrewAI**: Framework for orchestrating AI agents (data cleaner, analyzer, report generator)
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **ReportLab**: PDF generation
- **Google Gemini API**: Large language model for analytical insights

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Set up your API keys in the `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Run the main script:
```
python crew.py
```

2. Enter your information when prompted:
   - Your name (for report authorship)
   - Business question to analyze
   - Path to your dataset CSV file (or press Enter for default 'dataset.csv')

3. The system will process your data and generate a professional report in the `output` directory.

## Project Structure

- `crew.py`: Main application file containing the data analysis workflow
- `agents.py`: Definitions for the AI agents (data cleaner, analyzer, report generator)
- `tools.py`: Custom tools for data manipulation, visualization, and report generation
- `requirements.txt`: Dependencies for the project
- `.env`: Environment variables file for API keys
- `dataset.csv`: Sample dataset (metropolitan GDP data)
- `output/`: Directory where generated reports are saved
- `temp/`: Directory for temporary files during processing (can be safely cleaned after runs)

## Maintenance

- **Temp Directory**: After successful report generation, you can safely delete older files in the `temp/` directory to save disk space.

## Future Enhancements

- Additional visualization types
- More color themes for reports
- Interactive dashboard integration
- Multi-dataset comparative analysis
- Automated scheduled reporting

## About the Creator

The system was created by Manish Paneru, a Data Analyst passionate about using AI to automate repetitive analytical tasks. The goal is to leverage AI to handle routine data processing and report generation, allowing analysts to focus on high-value interpretative and strategic work.

---

*"Let AI do the heavy lifting, so you can focus on the insights that matter."* - Manish Paneru 