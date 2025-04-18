# Email Style Analyzer & Reply Generator

## Overview  
This project analyzes a sender's email writing style from the Enron Email Dataset and creates a JSON template for generating future replies in the same style. The system uses GPT (via Semantic Kernel in Python) to extract stylistic elements and stores them for consistent reply generation.

## Features  
- **Email Analysis**: Extracts writing style characteristics from a sender's emails  
- **Style Template Generation**: Creates a JSON template capturing:  
  - Tone (formal/informal)  
  - Common phrases  
  - Sign-off patterns  
  - Language level  
  - Style notes  
- **AI-Powered Reply Generation**: Uses the style template to generate context-appropriate replies matching the sender's style  

## Dataset  
We use the [Enron Email Dataset from Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset/data).

## How It Works  
1. **Email Extraction**: Filter emails from a specific sender  
2. **Style Analysis**: GPT analyzes each email's stylistic elements  
3. **Template Creation**: Compiles results into a JSON style template  
4. **Reply Generation**: Answers new emails using the template  

