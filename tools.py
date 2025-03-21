import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crewai_tools import FileWriterTool
from crewai.tools import BaseTool
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, ListFlowable, ListItem
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import numpy as np
import datetime
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import requests
from io import BytesIO
import matplotlib.font_manager as fm
import re

# Built-in tools
file_writer_tool = FileWriterTool()

# Register professional fonts for PDF and visualizations
fonts_registered = False
def register_fonts():
    """
    Register professional fonts for documents and visualizations.
    Falls back to standard fonts if custom fonts aren't available.
    """
    try:
        # For ReportLab PDF generation
        pdfmetrics.registerFont(TTFont('roboto', 'fonts/Roboto-Regular.ttf'))
        pdfmetrics.registerFont(TTFont('roboto-bold', 'fonts/Roboto-Bold.ttf'))
        pdfmetrics.registerFont(TTFont('roboto-italic', 'fonts/Roboto-Italic.ttf'))
        
        # Register font family
        registerFontFamily('roboto', normal='roboto', bold='roboto-bold', italic='roboto-italic')
        print("Professional fonts registered for reports")
    except:
        # If fonts aren't available or can't be registered, we'll use standard fonts
        print("Default fonts will be used for PDF reports")
    
    # Set matplotlib font settings for visualizations
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
    except:
        print("Default matplotlib fonts will be used for visualizations")

# Register fonts
register_fonts()

# Set default visualization style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Data Profile Tool
class DataProfileInput(BaseModel):
    """Input schema for DataProfileTool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dataframe: pd.DataFrame = Field(..., description="DataFrame to profile")
    include_stats: bool = Field(default=True, description="Whether to include statistical analysis")
    include_dtypes: bool = Field(default=True, description="Whether to include data types")
    include_unique_counts: bool = Field(default=True, description="Whether to include unique value counts")
    include_sample: bool = Field(default=True, description="Whether to include a sample of the data")
    include_correlations: bool = Field(default=True, description="Whether to include correlation analysis")

class DataProfileTool(BaseTool):
    name: str = "data_profiler"
    description: str = "Generates a comprehensive profile of the dataset including statistics, data types, and quality metrics"
    args_schema: Type[BaseModel] = DataProfileInput

    def _run(self, dataframe: pd.DataFrame, include_stats: bool = True, 
             include_dtypes: bool = True, include_unique_counts: bool = True,
             include_sample: bool = True, include_correlations: bool = True) -> dict:
        """Generate a comprehensive profile of the dataset."""
        profile = {
            'shape': dataframe.shape,
            'columns': dataframe.columns.tolist(),
            'missing_values': dataframe.isnull().sum().to_dict(),
            'missing_percent': {col: dataframe[col].isnull().mean() * 100 for col in dataframe.columns}
        }
        
        if include_dtypes:
            profile['dtypes'] = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
        
        if include_unique_counts:
            profile['unique_counts'] = {col: dataframe[col].nunique() for col in dataframe.columns}
            profile['unique_percent'] = {col: (dataframe[col].nunique() / len(dataframe)) * 100 for col in dataframe.columns}
        
        if include_stats:
            # For numeric columns
            num_cols = dataframe.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                profile['numeric_stats'] = dataframe[num_cols].describe().to_dict()
                # Add skewness and kurtosis
                profile['skewness'] = {col: dataframe[col].skew() for col in num_cols}
                profile['kurtosis'] = {col: dataframe[col].kurt() for col in num_cols}
            
            # For categorical columns
            cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                profile['categorical_stats'] = {}
                for col in cat_cols:
                    value_counts = dataframe[col].value_counts().head(10).to_dict()
                    profile['categorical_stats'][col] = value_counts
        
        if include_correlations and len(dataframe.select_dtypes(include=['number']).columns) > 1:
            profile['correlations'] = dataframe.select_dtypes(include=['number']).corr().to_dict()
        
        if include_sample:
            # Convert to string representation to avoid serialization issues
            sample_data = dataframe.head(5).to_dict(orient='records')
            profile['sample'] = sample_data
        
        return profile

# Enhanced Visualization Tool
class VisualizationInput(BaseModel):
    """Input schema for VisualizationTool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dataframe: pd.DataFrame = Field(..., description="DataFrame to visualize")
    plot_type: str = Field(..., description="Type of plot: line, bar, scatter, hist, heatmap, box, pie, area, violin, density, pair, count, joint, radar, waterfall, bubble, candlestick, gauge, or custom")
    x_column: Optional[str] = Field(default=None, description="Column for x-axis")
    y_column: Optional[str] = Field(default=None, description="Column for y-axis (optional)")
    title: str = Field(default="", description="Plot title")
    subtitle: Optional[str] = Field(default=None, description="Plot subtitle")
    caption: str = Field(default="", description="Detailed caption explaining the visualization")
    output_dir: str = Field(default="temp", description="Directory to save the visualization")
    additional_params: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters for the plot")
    hue: Optional[str] = Field(default=None, description="Grouping variable for color encoding")
    palette: Optional[str] = Field(default=None, description="Color palette to use")
    custom_code: Optional[str] = Field(default=None, description="Custom Python code to create a visualization (use with plot_type='custom')")

class VisualizationTool(BaseTool):
    name: str = "visualizer"
    description: str = "Creates professional data visualizations using matplotlib and seaborn with multiple plot types and customization options"
    args_schema: Type[BaseModel] = VisualizationInput

    def __init__(self):
        """Initialize the visualization tool"""
        super().__init__()
        plt.style.use('seaborn-v0_8-whitegrid')  # Use a professional style
        sns.set_context("talk")  # Larger context for better readability
        # Register professional fonts
        register_fonts()
        
    def _run(self, dataframe, plot_type="line", x_column=None, y_column=None, title=None, 
             caption=None, output_dir=".", hue=None, palette="viridis", figsize=(12, 8), 
             tick_rotation=45, grid=True, output_filename=None, additional_params=None):
        """
        Create and save a visualization.
        
        Args:
            dataframe: The pandas DataFrame
            plot_type: Type of plot to create
            x_column: Column to use for x-axis
            y_column: Column to use for y-axis  
            title: Title for the plot
            caption: Caption explaining the visualization
            output_dir: Directory to save the visualization
            hue: Column to use for grouping/coloring
            palette: Color palette to use
            figsize: Figure size (width, height) in inches
            tick_rotation: Rotation angle for x-axis tick labels
            grid: Whether to show grid lines
            output_filename: Custom filename to use (optional)
            additional_params: Dictionary of additional parameters specific to the plot type
            
        Returns:
            Path to the saved visualization
        """
        # Set default values for parameters not provided
        additional_params = additional_params or {}
        
        # Create figure and axes
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # Extract additional parameters
        bins = additional_params.get('bins', 30)
        kde = additional_params.get('kde', False)
        trendline = additional_params.get('trendline', False)
        
        # Create the requested plot type
        if plot_type == "line":
            sns.lineplot(data=dataframe, x=x_column, y=y_column, hue=hue, palette=palette)
        elif plot_type == "bar":
            ax = sns.barplot(data=dataframe, x=x_column, y=y_column, hue=hue, palette=palette)
        elif plot_type == "scatter":
            sns.scatterplot(data=dataframe, x=x_column, y=y_column, hue=hue, palette=palette)
            if trendline:
                # Add trend line if requested
                sns.regplot(data=dataframe, x=x_column, y=y_column, scatter=False, line_kws={"color": "red"})
        elif plot_type == "hist":
            sns.histplot(data=dataframe, x=x_column, bins=bins, kde=kde, hue=hue, palette=palette)
        elif plot_type == "box":
            sns.boxplot(data=dataframe, x=x_column, y=y_column, hue=hue, palette=palette)
        elif plot_type == "violin":
            sns.violinplot(data=dataframe, x=x_column, y=y_column, hue=hue, palette=palette)
        elif plot_type == "heatmap":
            mask = np.triu(np.ones_like(dataframe.corr(), dtype=bool))
            sns.heatmap(dataframe.corr(), mask=mask, cmap=palette, annot=True, fmt=".2f", 
                        linewidths=0.5, ax=ax, cbar_kws={"shrink": .8})
        elif plot_type == "pie":
            dataframe[x_column].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette(palette))
        elif plot_type == "area":
            dataframe.plot.area(x=x_column, stacked=True, ax=ax, colormap=palette)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        # Add title and labels
        if title:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Format axes
        plt.xticks(rotation=tick_rotation)
        if grid:
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
        # Add caption as figure suptitle (smaller text below the main plot)
        if caption:
            plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', 
                      fontsize=10, fontstyle='italic')
            # Add extra space at bottom for caption
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        else:
            plt.tight_layout()
            
        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename if not provided
        if output_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            x_col_str = f"_{x_column}" if x_column else ""
            y_col_str = f"_{y_column}" if y_column else ""
            output_filename = f"{plot_type}{x_col_str}{y_col_str}_{timestamp}.png"
            
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        return output_path

# DuckDuckGo Search Tool for internet information
class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = "Search the web for information using DuckDuckGo"
    
    def _run(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search DuckDuckGo for the given query and return results."""
        try:
            # URL for the DuckDuckGo API (this is a simplified version)
            base_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            results = []
            # Extract abstract
            if data.get('Abstract'):
                results.append({
                    'type': 'abstract',
                    'text': data.get('Abstract'),
                    'source': data.get('AbstractSource', 'Unknown')
                })
            
            # Extract related topics
            if data.get('RelatedTopics'):
                for topic in data.get('RelatedTopics')[:num_results]:
                    if 'Text' in topic:
                        results.append({
                            'type': 'related',
                            'text': topic.get('Text', ''),
                            'url': topic.get('FirstURL', '')
                        })
            
            return results
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

# Enhanced PDF Report Tool
class PDFReportInput(BaseModel):
    """Input schema for PDFReportTool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    content: Dict[str, str] = Field(..., description="Report content by section")
    title: str = Field(default="Analysis Report", description="Report title")
    business_question: str = Field(..., description="Business question being answered")
    visualizations: List[Dict[str, str]] = Field(default=[], description="List of visualizations with paths and captions")
    output_dir: str = Field(default="output", description="Directory to save the PDF report")
    executive_summary: Optional[str] = Field(default=None, description="Executive summary of the report")
    author: Optional[str] = Field(default=None, description="Author of the report")
    recommendations: Optional[List[str]] = Field(default=None, description="List of recommendations based on the analysis")
    appendix_content: Optional[Dict[str, str]] = Field(default=None, description="Additional content for the appendix")
    company_logo: Optional[str] = Field(default=None, description="Path to company logo image")

class PDFReportTool(BaseTool):
    name: str = "pdf_reporter"
    description: str = "Generates professional PDF reports with text, visualizations, and proper formatting"
    args_schema: Type[BaseModel] = PDFReportInput

    def __init__(self):
        """Initialize the PDF report tool"""
        super().__init__()
        # Register professional fonts for the report
        register_fonts()
    
    def _color_to_hex(self, color):
        """Convert a ReportLab color to hex format"""
        if hasattr(color, 'rgb'):
            r, g, b = color.rgb()
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        return '#000000'  # Default to black if conversion fails
    
    def _run(self, content, title, business_question=None, visualizations=None, 
            output_dir=".", executive_summary=None, author=None, date=None, 
            recommendations=None, appendix_content=None, use_standard_fonts=False,
            color_theme="blue"):
        """
        Generate a professional PDF report.
        
        Args:
            content: Dictionary with report content sections
            title: Report title
            business_question: The business question being addressed
            visualizations: List of dictionaries with visualization paths and captions
            output_dir: Directory to save the report
            executive_summary: Executive summary text
            author: Author name
            date: Report date (defaults to current date)
            recommendations: List of recommendations
            appendix_content: Additional content for the appendix
            use_standard_fonts: Whether to use standard fonts instead of custom fonts
            color_theme: Color theme for the report (blue, green, burgundy, gray)
            
        Returns:
            Path to the generated PDF report
        """
        # Set up PDF styles
        styles = getSampleStyleSheet()
        
        # Define color themes
        color_themes = {
            "blue": {
                "primary": colors.HexColor('#1a5276'),
                "secondary": colors.HexColor('#d4e6f1'),
                "accent": colors.HexColor('#3498db'),
                "text": colors.HexColor('#2c3e50')
            },
            "green": {
                "primary": colors.HexColor('#186a3b'),
                "secondary": colors.HexColor('#d4efdf'),
                "accent": colors.HexColor('#2ecc71'),
                "text": colors.HexColor('#1c2833')
            },
            "burgundy": {
                "primary": colors.HexColor('#76203a'),
                "secondary": colors.HexColor('#f2d7d5'),
                "accent": colors.HexColor('#c0392b'),
                "text": colors.HexColor('#1c2833')
            },
            "gray": {
                "primary": colors.HexColor('#424949'),
                "secondary": colors.HexColor('#eaeded'),
                "accent": colors.HexColor('#7f8c8d'),
                "text": colors.HexColor('#17202a')
            }
        }
        
        # Default to blue theme if specified theme not found
        theme = color_themes.get(color_theme.lower(), color_themes["blue"])
        
        # Set up fonts based on preference
        if use_standard_fonts:
            # Enhanced standard styles
            title_style = ParagraphStyle(
                'CustomTitle',
                fontName='Helvetica-Bold',
                fontSize=24,
                leading=30,
                alignment=1,  # Center
                spaceAfter=20,
                textColor=theme["primary"]
            )
            
            heading1_style = ParagraphStyle(
                'CustomHeading1',
                fontName='Helvetica-Bold',
                fontSize=16,
                leading=20,
                spaceBefore=16,
                spaceAfter=10,
                textColor=theme["primary"]
            )
            
            heading2_style = ParagraphStyle(
                'CustomHeading2',
                fontName='Helvetica-Bold',
                fontSize=14,
                leading=18,
                spaceBefore=14,
                spaceAfter=8,
                textColor=theme["primary"]
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                fontName='Helvetica',
                fontSize=10,
                leading=14,
                spaceBefore=6,
                spaceAfter=6,
                textColor=theme["text"]
            )
            
            caption_style = ParagraphStyle(
                'CustomCaption',
                fontName='Helvetica',
                fontSize=9,
                leading=12,
                alignment=1,  # Center
                textColor=theme["text"]
            )
            
            toc_heading_style = ParagraphStyle(
                'TOCHeading',
                fontName='Helvetica-Bold',
                fontSize=12,
                leading=16,
                textColor=theme["primary"],
                leftIndent=0
            )
            
            toc_item_style = ParagraphStyle(
                'TOCItem',
                fontName='Helvetica',
                fontSize=10,
                leading=14,
                textColor=theme["text"],
                leftIndent=20
            )
            
            summary_style = ParagraphStyle(
                'Summary',
                fontName='Helvetica',
                fontSize=11,
                leading=15,
                spaceBefore=8,
                spaceAfter=8,
                borderWidth=1,
                borderColor=theme["accent"],
                borderPadding=8,
                borderRadius=5,
                backColor=theme["secondary"],
                textColor=theme["text"]
            )
            
        else:
            # Custom styles with Roboto font
            title_style = ParagraphStyle(
                'CustomTitle',
                fontName='roboto-bold',
                fontSize=24,
                leading=30,
                alignment=1,  # Center
                spaceAfter=20,
                textColor=theme["primary"]
            )
            
            heading1_style = ParagraphStyle(
                'CustomHeading1',
                fontName='roboto-bold',
                fontSize=16,
                leading=20,
                spaceBefore=16,
                spaceAfter=10,
                textColor=theme["primary"]
            )
            
            heading2_style = ParagraphStyle(
                'CustomHeading2',
                fontName='roboto',
                fontSize=14,
                leading=18,
                spaceBefore=14,
                spaceAfter=8,
                textColor=theme["primary"],
                fontWeight='bold'
            )
            
            normal_style = ParagraphStyle(
                'CustomNormal',
                fontName='roboto',
                fontSize=10,
                leading=14,
                spaceBefore=6,
                spaceAfter=6,
                textColor=theme["text"]
            )
            
            caption_style = ParagraphStyle(
                'CustomCaption',
                fontName='roboto-italic',
                fontSize=9,
                leading=12,
                alignment=1,  # Center
                textColor=theme["text"]
            )
            
            toc_heading_style = ParagraphStyle(
                'TOCHeading',
                fontName='roboto-bold',
                fontSize=12,
                leading=16,
                textColor=theme["primary"],
                leftIndent=0
            )
            
            toc_item_style = ParagraphStyle(
                'TOCItem',
                fontName='roboto',
                fontSize=10,
                leading=14,
                textColor=theme["text"],
                leftIndent=20
            )
            
            summary_style = ParagraphStyle(
                'Summary',
                fontName='roboto',
                fontSize=11,
                leading=15,
                spaceBefore=8,
                spaceAfter=8,
                borderWidth=1,
                borderColor=theme["accent"],
                borderPadding=8,
                borderRadius=5,
                backColor=theme["secondary"],
                textColor=theme["text"]
            )
        
        # Create document
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        report_filename = f"professional_report_{timestamp}.pdf"
        report_path = os.path.join(output_dir, report_filename)
        
        doc = SimpleDocTemplate(
            report_path, 
            pagesize=letter,
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch, 
            topMargin=1*inch, 
            bottomMargin=1*inch
        )
        
        # Create story (content)
        story = []
        
        # ----- COVER PAGE -----
        # Add decorative top border element
        top_border_data = [
            ['']*3
        ]
        top_border = Table(top_border_data, colWidths=[doc.width/3]*3, rowHeights=[0.25*inch])
        top_border.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), theme["primary"]),
            ('BACKGROUND', (1, 0), (1, 0), theme["accent"]),
            ('BACKGROUND', (2, 0), (2, 0), theme["primary"]),
        ]))
        story.append(top_border)
        story.append(Spacer(1, 1*inch))
        
        # Add company logo if provided
        if appendix_content and 'company_logo' in appendix_content and os.path.exists(appendix_content.get('company_logo')):
            logo_img = Image(appendix_content.get('company_logo'), width=2.5*inch, height=1*inch)
            story.append(logo_img)
            story.append(Spacer(1, 0.5*inch))
        
        # Add title with enhanced styling
        title_text = f'<font color="{self._color_to_hex(theme["primary"])}">{title}</font>'
        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Add business question with professional formatting
        if business_question:
            question_text = f'<font size="12">Business Question:</font><br/><br/>{business_question}'
            story.append(Paragraph(question_text, normal_style))
            story.append(Spacer(1, 0.5*inch))
        
        # Add current date and author information
        current_date = date or datetime.datetime.now().strftime("%B %d, %Y")
        metadata_table_data = [
            ["Date:", current_date],
            ["Prepared by:", author or "AI Data Analyst"]
        ]
        metadata_table = Table(metadata_table_data, colWidths=[1.2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold' if use_standard_fonts else 'roboto-bold'),
            ('FONT', (1, 0), (1, -1), 'Helvetica' if use_standard_fonts else 'roboto'),
            ('TEXTCOLOR', (0, 0), (-1, -1), theme["text"]),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 1*inch))
        
        # Add executive summary if provided 
        if executive_summary:
            summary_title = Paragraph("Executive Summary", heading1_style)
            story.append(summary_title)
            story.append(Spacer(1, 0.1*inch))
            
            # Add decorative line under executive summary title
            summary_line_data = [['']]
            summary_line = Table(summary_line_data, colWidths=[3*inch], rowHeights=[1])
            summary_line.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), theme["accent"]),
                ('LINEBELOW', (0, 0), (0, 0), 1, theme["accent"]),
            ]))
            story.append(summary_line)
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph(executive_summary, summary_style))
        
        # Add bottom decorative border
        story.append(Spacer(1, 1*inch))
        bottom_border_data = [
            ['']*3
        ]
        bottom_border = Table(bottom_border_data, colWidths=[doc.width/3]*3, rowHeights=[0.25*inch])
        bottom_border.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), theme["primary"]),
            ('BACKGROUND', (1, 0), (1, 0), theme["accent"]),
            ('BACKGROUND', (2, 0), (2, 0), theme["primary"]),
        ]))
        story.append(bottom_border)
        
        # Add page break after cover
        story.append(PageBreak())
        
        # ----- TABLE OF CONTENTS -----
        story.append(Paragraph("Table of Contents", heading1_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Add decorative line under TOC title
        toc_line_data = [['']]
        toc_line = Table(toc_line_data, colWidths=[3*inch], rowHeights=[1])
        toc_line.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), theme["accent"]),
            ('LINEBELOW', (0, 0), (0, 0), 1, theme["accent"]),
        ]))
        story.append(toc_line)
        story.append(Spacer(1, 0.2*inch))
        
        # TOC content
        toc_data = []
        section_num = 1
        
        # Add standard sections from content dictionary
        for section, section_content in content.items():
            if section_content.strip():  # Only include non-empty sections
                section_title = section.replace('_', ' ').title()
                toc_data.append([f"{section_num}.", section_title, ""])
                section_num += 1
        
        # Add visualization section if there are visualizations
        if visualizations:
            toc_data.append([f"{section_num}.", "Visualizations", ""])
            section_num += 1
        
        # Add recommendations section if provided
        if recommendations:
            toc_data.append([f"{section_num}.", "Recommendations", ""])
            section_num += 1
        
        # Add appendix if provided
        if appendix_content:
            toc_data.append([f"{section_num}.", "Appendix", ""])
        
        # Create TOC table
        toc_table = Table(toc_data, colWidths=[0.4*inch, 4.5*inch, 0.4*inch])
        toc_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold' if use_standard_fonts else 'roboto-bold'),
            ('FONT', (1, 0), (1, -1), 'Helvetica' if use_standard_fonts else 'roboto'),
            ('TEXTCOLOR', (0, 0), (-1, -1), theme["text"]),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, theme["primary"]),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [theme["secondary"], colors.white]),
        ]))
        story.append(toc_table)
        
        story.append(PageBreak())
        
        # ----- MAIN CONTENT -----
        section_num = 1
        
        # Add main content sections
        for section, section_content in content.items():
            if section_content.strip():  # Only include non-empty sections
                section_title = section.replace('_', ' ').title()
                # Section title with number
                story.append(Paragraph(f"{section_num}. {section_title}", heading1_style))
                section_num += 1
                
                # Add decorative line under section title
                section_line_data = [['']]
                section_line = Table(section_line_data, colWidths=[3*inch], rowHeights=[1])
                section_line.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, 0), theme["accent"]),
                    ('LINEBELOW', (0, 0), (0, 0), 1, theme["accent"]),
                ]))
                story.append(section_line)
                story.append(Spacer(1, 0.2*inch))
                
                # Split the content by newlines and create paragraphs
                paragraphs = section_content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Check if this is a subsection heading (starts with ##)
                        if para.strip().startswith('##'):
                            subsection_title = para.strip().lstrip('#').strip()
                            story.append(Paragraph(subsection_title, heading2_style))
                        else:
                            # Process lists (lines starting with * or -)
                            lines = para.split('\n')
                            list_items = []
                            
                            # Check if this is a list
                            if any(line.strip().startswith(('*', '-', '•')) for line in lines):
                                for line in lines:
                                    if line.strip().startswith(('*', '-', '•')):
                                        item_text = line.strip().lstrip('*-• ').strip()
                                        list_items.append(ListItem(Paragraph(item_text, normal_style)))
                                
                                if list_items:
                                    story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=20, 
                                                             bulletColor=theme["accent"], start=None))
                            else:
                                # Regular paragraph - enhance first letter of paragraphs
                                if len(para) > 1:
                                    first_char = para[0]
                                    rest_of_para = para[1:]
                                    enhanced_para = f'<font color="{self._color_to_hex(theme["primary"])}" size="12"><b>{first_char}</b></font>{rest_of_para}'
                                    story.append(Paragraph(enhanced_para, normal_style))
                                else:
                                    story.append(Paragraph(para, normal_style))
                
                story.append(Spacer(1, 0.2*inch))
        
        # ----- VISUALIZATIONS -----
        if visualizations:
            story.append(Paragraph(f"{section_num}. Visualizations", heading1_style))
            section_num += 1
            
            # Add decorative line under visualization title
            viz_line_data = [['']]
            viz_line = Table(viz_line_data, colWidths=[3*inch], rowHeights=[1])
            viz_line.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), theme["accent"]),
                ('LINEBELOW', (0, 0), (0, 0), 1, theme["accent"]),
            ]))
            story.append(viz_line)
            story.append(Spacer(1, 0.2*inch))
            
            for i, viz in enumerate(visualizations):
                viz_path = viz.get('path', '')
                viz_caption = viz.get('caption', f"Figure {i+1}")
                viz_title = viz.get('title', '')
                
                if viz_path and os.path.exists(viz_path):
                    # Create a visualization container with border
                    viz_container_data = [['']]
                    if viz_title:
                        # Add title if provided
                        story.append(Paragraph(f"Figure {i+1}: {viz_title}", heading2_style))
                    
                    # Add the image with professional formatting
                    img = Image(viz_path, width=6.5*inch, height=4.5*inch)
                    
                    # Create a bordered container for the image
                    image_data = [[img]]
                    image_table = Table(image_data, colWidths=[6.5*inch])
                    image_table.setStyle(TableStyle([
                        ('BOX', (0, 0), (-1, -1), 0.5, theme["primary"]),
                        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('PADDING', (0, 0), (-1, -1), 10),
                    ]))
                    story.append(image_table)
                    
                    # Add caption with enhanced styling
                    caption_with_style = f'<i>{viz_caption}</i>'
                    story.append(Paragraph(caption_with_style, caption_style))
                    story.append(Spacer(1, 0.3*inch))
        
        # ----- RECOMMENDATIONS -----
        if recommendations:
            story.append(Paragraph(f"{section_num}. Recommendations", heading1_style))
            section_num += 1
            
            # Add decorative line under recommendations title
            rec_line_data = [['']]
            rec_line = Table(rec_line_data, colWidths=[3*inch], rowHeights=[1])
            rec_line.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), theme["accent"]),
                ('LINEBELOW', (0, 0), (0, 0), 1, theme["accent"]),
            ]))
            story.append(rec_line)
            story.append(Spacer(1, 0.2*inch))
            
            # Create a professional-looking recommendations box
            rec_content = []
            for i, rec in enumerate(recommendations):
                rec_content.append([f"{i+1}.", rec])
            
            # Format recommendations as a styled table
            if rec_content:
                rec_table = Table(rec_content, colWidths=[0.4*inch, 6*inch])
                rec_table.setStyle(TableStyle([
                    ('FONT', (0, 0), (0, -1), 'Helvetica-Bold' if use_standard_fonts else 'roboto-bold'),
                    ('FONT', (1, 0), (1, -1), 'Helvetica' if use_standard_fonts else 'roboto'),
                    ('TEXTCOLOR', (0, 0), (0, -1), theme["primary"]),
                    ('TEXTCOLOR', (1, 0), (1, -1), theme["text"]),
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BACKGROUND', (0, 0), (-1, -1), theme["secondary"]),
                    ('BOX', (0, 0), (-1, -1), 0.5, theme["primary"]),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ]))
                story.append(rec_table)
            
            story.append(Spacer(1, 0.2*inch))
        
        # ----- APPENDIX -----
        if appendix_content:
            story.append(PageBreak())
            story.append(Paragraph(f"{section_num}. Appendix", heading1_style))
            
            # Add decorative line under appendix title
            appendix_line_data = [['']]
            appendix_line = Table(appendix_line_data, colWidths=[3*inch], rowHeights=[1])
            appendix_line.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), theme["accent"]),
                ('LINEBELOW', (0, 0), (0, 0), 1, theme["accent"]),
            ]))
            story.append(appendix_line)
            story.append(Spacer(1, 0.2*inch))
            
            appendix_counter = 1
            for section, section_content in appendix_content.items():
                if section != 'company_logo':  # Skip logo entry
                    section_title = section.replace('_', ' ').title()
                    story.append(Paragraph(f"Appendix {appendix_counter}: {section_title}", heading2_style))
                    story.append(Paragraph(section_content, normal_style))
                    story.append(Spacer(1, 0.2*inch))
                    appendix_counter += 1
        
        # Add footer with page numbers and header with report title
        def add_page_number_and_header(canvas, doc):
            canvas.saveState()
            
            # Header
            canvas.setFillColor(theme["primary"])
            canvas.setFont('Helvetica-Bold' if use_standard_fonts else 'roboto-bold', 9)
            canvas.drawString(doc.leftMargin, doc.height + doc.topMargin - 0.25*inch, title)
            
            # Header line
            canvas.setStrokeColor(theme["accent"])
            canvas.line(doc.leftMargin, doc.height + doc.topMargin - 0.3*inch, 
                       doc.width + doc.leftMargin, doc.height + doc.topMargin - 0.3*inch)
            
            # Footer line
            canvas.setStrokeColor(theme["accent"])
            canvas.line(doc.leftMargin, doc.bottomMargin - 0.3*inch, 
                       doc.width + doc.leftMargin, doc.bottomMargin - 0.3*inch)
            
            # Footer with page number
            page_num = f"Page {canvas.getPageNumber()}"
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            
            # Left side: date
            canvas.setFont('Helvetica' if use_standard_fonts else 'roboto', 8)
            canvas.setFillColor(theme["text"])
            canvas.drawString(doc.leftMargin, doc.bottomMargin - 0.5*inch, current_date)
            
            # Center: page number
            canvas.drawCentredString(doc.width/2 + doc.leftMargin, doc.bottomMargin - 0.5*inch, page_num)
            
            # Right side: prepared by
            if author:
                canvas.drawRightString(doc.width + doc.leftMargin, doc.bottomMargin - 0.5*inch, f"Prepared by: {author}")
            
            canvas.restoreState()
        
        # Build PDF with enhanced headers and footers
        doc.build(story, onFirstPage=add_page_number_and_header, onLaterPages=add_page_number_and_header)
        
        return report_path

# Initialize tools
data_profile_tool = DataProfileTool()
visualization_tool = VisualizationTool()
pdf_report_tool = PDFReportTool()
duckduckgo_search_tool = DuckDuckGoSearchTool()

 







