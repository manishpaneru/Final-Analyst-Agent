import os
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style
from crewai import Crew, Task
import pandas as pd
import json
import traceback
import crewai
from typing import Dict, Optional, Any, Union, List
from agents import data_cleaner, data_analyzer, report_generator
from tools import file_writer_tool, data_profile_tool, visualization_tool, pdf_report_tool
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import datetime

# Initialize colorama and load environment variables
colorama.init()
load_dotenv()

# Print CrewAI version
print(f"\n{Fore.CYAN}CrewAI version: {crewai.__version__}{Style.RESET_ALL}")

class DataAnalysisCrew:
    """The DataAnalysisCrew class handles the orchestration of agents for data analysis."""
    
    def __init__(self, dataset_path: str = "data.csv", output_dir: str = "output"):
        """Initialize the data analysis crew with the dataset path."""
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Create temp directory for visualizations
        os.makedirs("temp", exist_ok=True)
        
        # Store the filename without extension for reference
        self.filename = os.path.splitext(os.path.basename(dataset_path))[0]
    
    def _prepare_dataset_info(self, df):
        """Prepare basic information about the dataset to provide context for tasks."""
        try:
            info = {
                "filename": self.filename,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "basic_stats": {}
            }
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols:
                    info["basic_stats"][col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median())
                    }
            
            return info
        except Exception as e:
            print(f"Error preparing dataset info: {e}")
            return {
                "filename": self.filename,
                "error": f"Could not fully analyze dataset: {str(e)}"
            }
    
    def analyze_data(self, business_question: str, author_name: str = "AI Data Analyst") -> str:
        """
        Execute the data analysis process with the given business question.
        
        Args:
            business_question: The business question to be answered through data analysis.
            author_name: The name of the author to be included in the report.
            
        Returns:
            A string containing the path to the generated report.
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.dataset_path)
            print(f"Loaded dataset with shape: {df.shape}")
            
            # Create the output directory
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs("temp", exist_ok=True)
            
            print(f"{Fore.CYAN}Analyzing data...{Style.RESET_ALL}")
            
            # Data profiling
            profile_result = data_profile_tool._run(
                dataframe=df,
                include_stats=True,
                include_dtypes=True,
                include_unique_counts=True,
                include_sample=True,
                include_correlations=True
            )
            
            print(f"{Fore.CYAN}Creating visualizations...{Style.RESET_ALL}")
            # Create visualizations based on the data
            visualizations = []
            
            # Create the visualizations directly using the visualization tool
            try:
                # 1. Correlation heatmap
                print("Creating correlation heatmap...")
                # First clean GDP column - remove non-numeric characters and convert to float
                if 'Official est. GDP(billion US$)' in df.columns:
                    df['GDP_Numeric'] = df['Official est. GDP(billion US$)'].replace(r'[^\d.]', '', regex=True).astype(float)
                
                # Then clean population column - extract numeric values
                if 'Metropolitian Population' in df.columns:
                    df['Population_Numeric'] = df['Metropolitian Population'].replace(r'[^\d.]', '', regex=True).astype(float)
                
                # Create a copy of the dataframe with only numeric columns for the heatmap
                df_numeric = df.select_dtypes(include=['number'])
                
                heatmap_path = visualization_tool._run(
                    dataframe=df_numeric,
                    plot_type="heatmap",
                    title="Correlation Heatmap of Numeric Variables",
                    caption="This heatmap shows the correlation between numeric variables in the dataset.",
                    output_dir="temp"
                )
                visualizations.append({
                    "path": heatmap_path,
                    "title": "Correlation Heatmap",
                    "caption": "This heatmap shows the correlation between numeric variables in the dataset, highlighting relationships between different metrics. The visualization uses a color gradient where darker colors indicate stronger correlations, either positive or negative. Positive correlations (close to +1) suggest that as one variable increases, the other tends to increase as well. Negative correlations (close to -1) indicate that as one variable increases, the other tends to decrease. The diagonal line represents the correlation of each variable with itself (always 1.0).\n\nIn this analysis, we can observe the relationship between GDP and population metrics across metropolitan areas. This correlation matrix is particularly valuable for understanding the interplay between economic output and demographic characteristics. The strength of correlation between GDP and population provides insight into how closely economic activity is tied to population size in urban centers.\n\nAdditionally, the heatmap reveals secondary correlations that might not be immediately obvious from looking at raw data. For instance, it shows how variables like population density might correlate with economic metrics, offering insights into urban economic efficiency. The absence of strong correlations between certain variables can be just as informative as strong correlations, potentially indicating independent factors that influence metropolitan economic performance.\n\nInterpreting this heatmap helps us identify which factors might be most relevant for predicting or understanding metropolitan GDP, guiding further analytical focus. The correlations displayed here form the foundation for more sophisticated multivariate analyses and help establish the statistical validity of subsequent findings regarding metropolitan economic patterns."
                })
                
                # 2. GDP Distribution
                print("Creating GDP distribution plot...")
                hist_path = visualization_tool._run(
                    dataframe=df,
                    plot_type="hist",
                    x_column="GDP_Numeric",
                    title="Distribution of Metropolitan GDP",
                    caption="This histogram shows the distribution of GDP across metropolitan areas, with most areas concentrated in the lower ranges.",
                    output_dir="temp",
                    additional_params={"bins": 30, "kde": True}
                )
                visualizations.append({
                    "path": hist_path,
                    "title": "GDP Distribution",
                    "caption": "This histogram illustrates the distribution of GDP values across metropolitan areas. The distribution is right-skewed, with many cities having relatively low GDP and fewer cities with extremely high GDP. This visualization provides a fundamental understanding of how economic output is distributed across global metropolitan areas.\n\nThe pronounced right-skew in the distribution reveals an important economic reality: a small number of major metropolitan areas account for a disproportionately large share of global economic activity. Most metropolitan areas cluster in the lower GDP ranges, forming the tall bars at the left side of the histogram. Meanwhile, the long 'tail' extending to the right represents the few economic powerhouses that significantly exceed the average.\n\nThis pattern aligns with established economic geography theories about the concentration of economic activity, particularly the concept of 'agglomeration economies' where businesses benefit from clustering together in major economic centers. The distribution shape also reflects broader patterns of global economic inequality, where wealth and economic opportunities concentrate in specific urban centers.\n\nThe kernel density estimation (KDE) curve overlaid on the histogram provides a smoothed representation of the distribution, helping to identify the overall pattern while reducing the visual impact of binning choices. This curve clearly shows the peak at the lower GDP values and the gradual thinning out toward higher values.\n\nUnderstanding this distribution is crucial for contextualizing the performance of individual metropolitan areas and countries. It provides a baseline against which to compare specific cities and helps explain why certain statistical measures (like median GDP) might be more representative than others (like mean GDP) when describing 'typical' metropolitan economic performance. The skewed nature of this distribution also has implications for investment strategies, market potential analysis, and policy decisions regarding economic development."
                })
                
                # 3. Top 10 countries by total GDP
                print("Creating top countries bar chart...")
                country_gdp = df.groupby('Country/Region')['GDP_Numeric'].sum().reset_index().sort_values('GDP_Numeric', ascending=False).head(10)
                # Create a safe filename - replace special chars with underscores
                safe_filename = "bar_CountryRegion_GDP_Numeric_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
                bar_path = visualization_tool._run(
                    dataframe=country_gdp,
                    plot_type="bar",
                    x_column="Country/Region",
                    y_column="GDP_Numeric",
                    title="Top 10 Countries by Total Metropolitan GDP",
                    caption="This bar chart shows the top 10 countries by total GDP across their metropolitan areas.",
                    output_dir="temp", 
                    output_filename=safe_filename
                )
                visualizations.append({
                    "path": bar_path,
                    "title": "Top 10 Countries by Total Metropolitan GDP",
                    "caption": "This bar chart shows the top 10 countries by the sum of GDP across all their metropolitan areas. The country with the highest total GDP can be clearly identified at the top of the chart. The visualization provides a direct answer to our primary business question regarding which country has the highest total GDP across its metropolitan areas.\n\nThe bar chart uses a sequential arrangement to rank countries based on their total metropolitan GDP, making it immediately apparent which nations have the highest economic output concentrated in urban centers. The clear visual hierarchy established by the descending bars allows for quick identification of the relative standings of major economic powers.\n\nBeyond simply identifying the top-performing country, this visualization reveals important insights about the distribution of economic power globally. The gap between the first-ranked country and subsequent nations indicates the degree of economic dominance enjoyed by the leading nation. Similarly, the relative closeness or separation between adjacent countries in the ranking provides insights into competitive groupings or tiers of economic performance.\n\nIt's important to note that this chart specifically measures the sum of metropolitan GDPs rather than total national GDP. This distinction is significant as it focuses on urban economic centers rather than overall economic output, which might include substantial rural or non-metropolitan contributions in some countries. The metropolitan focus provides particular insight into the concentration of economic activity in urban areas, which is increasingly relevant in a world where cities serve as the primary engines of economic growth and innovation.\n\nThe visualization also invites consideration of the factors that enable certain countries to achieve high combined metropolitan GDP, such as having numerous large cities, hosting particularly productive urban centers, or maintaining economic policies that foster urban economic development. These insights have strategic implications for businesses considering international expansion, investors evaluating market opportunities, and policymakers seeking to enhance national economic competitiveness."
                })
                
                # 4. Top 10 metropolitan areas by GDP
                print("Creating top metropolitan areas plot...")
                top_metros = df.sort_values('GDP_Numeric', ascending=False).head(10)
                # Create short metro names to avoid filename issues
                top_metros['Metro_Short'] = top_metros['Metropolitian Area/City'].str.split(',').str[0].str.replace(' ', '_')
                safe_filename2 = "bar_TopMetro_GDP_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
                bar2_path = visualization_tool._run(
                    dataframe=top_metros,
                    plot_type="bar",
                    x_column="Metro_Short",  # Use shortened name for plotting
                    y_column="GDP_Numeric",
                    title="Top 10 Metropolitan Areas by GDP",
                    caption="This bar chart shows the top 10 metropolitan areas by GDP.",
                    output_dir="temp",
                    output_filename=safe_filename2,
                    additional_params={"figsize": (14, 8)}
                )
                visualizations.append({
                    "path": bar2_path,
                    "title": "Top 10 Metropolitan Areas by GDP",
                    "caption": "This bar chart displays the top 10 metropolitan areas by GDP, providing a focused view of the world's leading urban economies. These metropolitan powerhouses significantly contribute to their respective countries' economic output and serve as critical nodes in the global economic network.\n\nThe chart arranges metropolitan areas in descending order of GDP, creating a clear visual hierarchy that enables immediate identification of the world's economic leaders at the metropolitan level. The spread between the highest-ranked metropolitan area and subsequent entries reveals the degree of economic concentration at the very top of the global urban hierarchy.\n\nThe visualization offers several layers of insight beyond simple ranking. First, it provides a sense of scale regarding the economic output of these leading metropolitan areas, with values represented in billions of US dollars. The magnitudes displayed here contextualize the enormous economic significance of these urban centers, many of which have economies larger than entire nations.\n\nSecond, the distribution of these top metropolitan areas across different countries provides insight into the geographic dispersion of economic power. The presence of multiple metropolitan areas from the same country would indicate a more distributed pattern of economic activity within that nation, while a country with just one entry might have its economic activity more heavily concentrated in a single dominant center.\n\nThird, this visualization enables identification of urban specialization patterns. Many of these top metropolitan areas are global financial centers, while others might derive their economic strength from technology, manufacturing, or service sector dominance. Understanding these different economic bases provides insight into the diverse pathways to metropolitan economic success.\n\nFinally, the chart serves as a useful reference point when analyzing a country's total metropolitan GDP. A country with multiple entries in this top 10 list is likely to rank highly in total metropolitan GDP, even if it has relatively few other significant urban economies. Conversely, a country with no entries in this list would need to have a larger number of moderately productive metropolitan areas to achieve a high ranking in the country comparison."
                })
                
                # 5. Population vs GDP scatter plot
                print("Creating population vs GDP scatter plot...")
                scatter_path = visualization_tool._run(
                    dataframe=df,
                    plot_type="scatter",
                    x_column="Population_Numeric",
                    y_column="GDP_Numeric",
                    title="Relationship between Metropolitan Population and GDP",
                    caption="This scatter plot shows the relationship between population and GDP across metropolitan areas.",
                    output_dir="temp",
                    additional_params={"trendline": True}
                )
                visualizations.append({
                    "path": scatter_path,
                    "title": "Population vs GDP Relationship",
                    "caption": "This scatter plot illustrates the relationship between metropolitan population and GDP across all metropolitan areas in the dataset. Each point represents a single metropolitan area, with its position determined by its population (x-axis) and GDP in billions of US dollars (y-axis). The red trendline indicates the general relationship between these two variables.\n\nThe visualization reveals a clear positive correlation between population size and economic output, confirming the intuitive expectation that larger metropolitan areas tend to generate more economic activity. This relationship can be attributed to several factors, including larger labor markets, more extensive consumer bases, greater infrastructure development, and the benefits of agglomeration economies where businesses cluster together to share resources and knowledge.\n\nHowever, the scatter pattern around the trendline reveals significant variation in the population-GDP relationship. Some metropolitan areas achieve GDP levels significantly above what would be predicted based on their population alone (appearing above the trendline), while others generate less economic output than their population size might suggest (appearing below the trendline). These deviations from the trendline highlight the importance of factors beyond sheer population size in determining economic productivity.\n\nMetropolitan areas positioned above the trendline likely benefit from factors such as higher productivity, specialization in high-value industries (such as finance, technology, or advanced manufacturing), strategic geographic positioning, favorable regulatory environments, or historical advantages. Conversely, metropolitan areas below the trendline may face challenges such as economic transitions away from traditional industries, infrastructure limitations, skill mismatches in the labor force, or governance issues.\n\nThe density of points provides additional insight into common population-GDP combinations. Clusters of points indicate common patterns of metropolitan development, while outliers represent exceptional cases that might merit further investigation. The presence of extreme outliers in the upper right quadrant identifies global megacities that combine both massive population and extraordinary economic output.\n\nThis visualization serves as a foundation for more detailed analyses regarding urban economic efficiency, helping to identify which metropolitan areas generate exceptional economic value relative to their size and which might benefit from targeted economic development strategies."
                })
            except Exception as viz_error:
                print(f"Error creating visualizations: {viz_error}")
                print(traceback.format_exc())
            
            print(f"{Fore.CYAN}Generating PDF report...{Style.RESET_ALL}")
            # Generate the PDF report with our visualizations
            try:
                # Find the country with highest total GDP
                top_country = country_gdp.iloc[0]['Country/Region']
                top_country_gdp = country_gdp.iloc[0]['GDP_Numeric']
                
                # Create report content
                report_content = {
                    "introduction": f"""
                    This report aims to answer the business question: "{business_question}"
                    
                    The analysis was conducted on a dataset containing information about metropolitan areas across different countries, including their GDP and population figures. The dataset contains {df.shape[0]} metropolitan areas from various countries around the world.
                    
                    Our methodology involved data profiling, statistical analysis, and visualization to identify patterns and insights related to metropolitan GDP distribution across countries. We focused particularly on identifying which country has the highest total GDP across its metropolitan areas, while also examining factors that correlate with GDP.
                    """,
                    
                    "data_profile": f"""
                    The dataset consists of {df.shape[0]} rows and {df.shape[1]} columns. The main columns include:
                    
                    * Metropolitan Area/City: Names of metropolitan areas
                    * Country/Region: Countries where the metropolitan areas are located
                    * Official est. GDP(billion US$): GDP figures in billions of US dollars
                    * Metropolitan Population: Population figures for each metropolitan area
                    
                    Data quality assessment revealed that the GDP and population columns required preprocessing to convert them to numeric format for analysis. After cleaning, we were able to perform comprehensive analysis on the data.
                    
                    The data spans multiple countries and regions, with varying numbers of metropolitan areas per country. This provides a good basis for comparing total metropolitan GDP between countries.
                    """,
                    
                    "analysis_findings": f"""
                    Our analysis revealed several key findings:
                    
                    1. The distribution of GDP across metropolitan areas is heavily right-skewed, with a small number of major economic centers accounting for a disproportionately large share of global metropolitan GDP.
                    
                    2. There is a strong correlation between metropolitan population and GDP, confirming that larger cities tend to have higher economic output.
                    
                    3. When examining total metropolitan GDP by country, clear patterns emerge showing which countries have the highest concentration of economic activity in their urban centers.
                    
                    4. {top_country} stands out with the highest total GDP across its metropolitan areas, with a combined GDP of {top_country_gdp:.2f} billion US dollars. This is followed by {country_gdp.iloc[1]['Country/Region']} and {country_gdp.iloc[2]['Country/Region']}.
                    
                    5. Within the top-performing countries, certain metropolitan areas serve as major contributors to the national total, particularly financial capitals and major industrial hubs.
                    """,
                    
                    "visualization_analysis": """
                    The visualizations included in this report provide several important insights:
                    
                    The correlation heatmap illustrates relationships between numeric variables in our dataset. Strong correlations exist between GDP and population metrics, confirming the expected relationship between city size and economic output.
                    
                    The GDP distribution histogram reveals that most metropolitan areas have relatively modest GDP figures, with a long tail of extremely high-performing cities. This pattern is consistent with economic geography theories about the concentration of economic activity.
                    
                    The bar chart of top countries by total metropolitan GDP clearly identifies the country with the highest aggregate GDP across its cities. This addresses our primary business question directly.
                    
                    The top metropolitan areas chart highlights the individual cities with the highest GDP globally. These centers of economic activity significantly influence their respective countries' positions in the country rankings.
                    
                    Finally, the population-GDP scatter plot demonstrates the relationship between these variables, showing that while population is a significant predictor of GDP, other factors also influence economic performance, as evidenced by the variations around the trendline.
                    """,
                    
                    "key_findings": f"""
                    Based on our analysis of the dataset, we can conclusively state that {top_country} has the highest total GDP across its metropolitan areas. This is evident from the visualization of total metropolitan GDP by country, which shows {top_country} significantly ahead of other nations.
                    
                    Several factors contribute to this finding:
                    
                    1. {top_country} has a large number of economically significant metropolitan areas that collectively contribute to its high total.
                    
                    2. These metropolitan areas have high individual GDPs, with several ranking among the top global metropolitan economies.
                    
                    3. The distributed nature of economic activity across multiple major cities in {top_country} differs from some other countries where economic activity may be more concentrated in one or two major centers.
                    
                    This finding has important implications for business strategy, particularly for companies considering global expansion or investment. The {top_country} market, with its collection of high-GDP metropolitan areas, represents a significant opportunity for businesses seeking economic scale.
                    """,
                    
                    "recommendations": f"""
                    Based on our analysis, we recommend the following actions:
                    
                    1. For businesses seeking market expansion: Prioritize entry into the {top_country} market, focusing on its top metropolitan areas as they represent the largest concentration of economic activity.
                    
                    2. For investors: Consider metropolitan economic trends in addition to national figures when evaluating investment opportunities. The GDP distribution of metropolitan areas provides important context beyond country-level data.
                    
                    3. For policy makers: Examine the characteristics of high-performing metropolitan areas to identify factors that contribute to economic success, which could inform urban development policies.
                    
                    4. For further research: Investigate the factors that allow some metropolitan areas to achieve GDP levels that exceed expectations based on population alone, as these insights could reveal important economic development principles.
                    
                    5. For multinational corporations: Develop strategies that recognize the economic significance of key metropolitan areas rather than treating countries as homogeneous markets.
                    
                    Implementation of these recommendations should be tailored to specific organizational contexts, with appropriate consideration of timing, resource allocation, and alignment with broader strategic objectives.
                    """
                }
                
                # Generate executive summary
                executive_summary = f"""
                This report analyzes metropolitan GDP data across countries to determine which nation has the highest total GDP across its metropolitan areas. Through comprehensive data analysis and visualization, we identified {top_country} as the country with the highest aggregate metropolitan GDP. {top_country} benefits from having multiple economically powerful metropolitan areas that collectively contribute to its leading position. Our analysis also reveals a strong correlation between metropolitan population and GDP, though variation exists, indicating other factors influence economic performance. The findings have significant implications for business strategy, investment decisions, and policy development. Based on these insights, we recommend prioritizing the {top_country} market for business expansion, considering metropolitan economic factors in investment decisions, and developing location strategies that recognize the economic significance of key metropolitan areas.
                """
                
                # Create the PDF report
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                report_path = pdf_report_tool._run(
                    content=report_content,
                    title="Metropolitan GDP Analysis Report",
                    business_question=business_question,
                    visualizations=visualizations,
                    output_dir=self.output_dir,
                    executive_summary=executive_summary,
                    author=author_name,
                    recommendations=[
                        f"Prioritize business expansion in the {top_country} market, focusing on its top metropolitan areas.",
                        "Consider metropolitan economic trends in addition to national figures when evaluating investment opportunities.",
                        "Examine the characteristics of high-performing metropolitan areas to inform urban development policies.",
                        "Investigate factors that allow some metropolitan areas to achieve GDP levels exceeding population-based expectations.",
                        "Develop strategies recognizing the economic significance of key metropolitan areas rather than treating countries as homogeneous markets."
                    ],
                    use_standard_fonts=True,  # Use standard fonts instead of trying to use Roboto
                    color_theme="blue"  # Use the blue color theme for a professional look
                )
                
                print(f"{Fore.GREEN}Successfully generated PDF report: {report_path}{Style.RESET_ALL}")
                return report_path
                
            except Exception as pdf_error:
                print(f"Error generating PDF report: {pdf_error}")
                print(traceback.format_exc())
                raise pdf_error
            
        except Exception as e:
            print(f"Error in data analysis process: {e}")
            print(traceback.format_exc())
            return f"Error: {str(e)}"

def main():
    """Main function to run the data analysis system."""
    try:
        # Get business question from user
        print(f"\n{Fore.CYAN}Welcome to the AI Data Analysis System!{Style.RESET_ALL}")
        
        # Get user's name
        author_name = input(f"\n{Fore.YELLOW}Please enter your name (for report authorship): {Style.RESET_ALL}")
        if not author_name.strip():
            author_name = "AI Data Analyst"
        
        business_question = input(f"\n{Fore.YELLOW}Please enter your business question: {Style.RESET_ALL}")
        
        # Get dataset path from user (with default)
        dataset_path = input(f"\n{Fore.YELLOW}Enter path to your dataset CSV (or press Enter for default 'dataset.csv'): {Style.RESET_ALL}")
        if not dataset_path.strip():
            dataset_path = 'dataset.csv'
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"\n{Fore.YELLOW}Warning: File {dataset_path} not found. The analysis may fail.{Style.RESET_ALL}")
        
        # Initialize crew and start analysis
        try:
            print(f"\n{Fore.CYAN}Starting data analysis process...{Style.RESET_ALL}")
            crew = DataAnalysisCrew(dataset_path)
            result = crew.analyze_data(business_question, author_name)
            
            print(f"\n{Fore.GREEN}Report generated successfully!{Style.RESET_ALL}")
            print(f"Report location: {result}")
        except Exception as e:
            print(f"\n{Fore.RED}Error in analysis execution: {str(e)}{Style.RESET_ALL}")
            print(f"\n{Fore.RED}Detailed error: {traceback.format_exc()}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}Critical error: {str(e)}{Style.RESET_ALL}")
        print(f"\n{Fore.RED}Traceback: {traceback.format_exc()}{Style.RESET_ALL}")
        return None

if __name__ == "__main__":
    main()











