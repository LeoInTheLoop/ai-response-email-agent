# Email Style Analyzer & Reply Generator

## 🧠 Overview  
This project analyzes a sender's email writing style using the Enron Email Dataset and generates a JSON-based template for consistent, AI-powered reply generation. It leverages GPT (via [Semantic Kernel](https://github.com/microsoft/semantic-kernel)) to extract stylistic patterns and apply them to future replies.

## ✨ Features  
- **Email Style Analysis**  
  Extracts writing characteristics from a sender’s emails.

- **Style Template Generation**  
  Automatically creates a JSON template with key stylistic elements:
  - `tone`: formal / informal  
  - `common_phrases`: frequently used phrases  
  - `sign_offs`: typical email closings  
  - `language_level`: e.g., native, intermediate  
  - `style_notes`: other stylistic observations

- **AI-Powered Reply Generation**  
  Uses the style template to craft new replies that match the sender’s style.

## 📁 Dataset  
We use the [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset/data) from Kaggle, which contains thousands of real business emails exchanged among Enron employees.

## ⚙️ How It Works  
1. **Email Extraction**  
   - Filter emails from a specific sender  
   - Collect a sample (e.g., ~100 emails) for style analysis  

2. **Style Analysis**  
   - Use GPT to extract tone, language level, patterns, and phrases  

3. **Template Creation**  
   - Compile results into a single JSON style profile  

4. **Reply Generation**  
   - Given a new email, load the style template and generate a reply in the sender’s style

## 🛠 To-Do  
- [ ] **Email Extraction**: Write a function to filter emails from one sender (~100 emails)  
- [ ] **Style Analysis**: Use GPT to analyze and output a JSON style template  
- [ ] **Agent Prompt Design**: Create a basic agent (e.g., `PoetAgent`) and define its prompt behavior  
- [ ] **Semantic Kernel Integration**: Register plugin that uses pre-generated style JSON to guide AI email reply generation

## ✅ Done  
- ✅ Created basic agent: `BasicAgent` in `basicAgent.py`


我们怎么运作
前端用add in ，basic button   总之最好能得到将要回复的信息，减少中间  像 对话框
后端flask 
  - 当第一次打开的 时候用history 训练并产生json （ 后期向量 rag）
  - 当单个邮件进来 调用json 返回

simmlar app ai email writer（gmail add on）