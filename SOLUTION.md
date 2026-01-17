# Neuberger Berman  
## Data Process and Insight Demo  

**Tony Wang**  
**Jan 2026**

## Answer to questions:

**Data processing: what issues did you find and how did you handle them?**
A: the main issue I found on the data is some transaction are missing the transaction the trader value. To handle this, I wrote data pre-processing pipeline to filter out rows missing value on any columns, and saved these rejected rows in a separate table in the database

**API design: how did you organize your code**
A: I designed the system so that the app splits into 5 isolate modules with as little inter-dependencies as possible. The module each handles: data storage (abstraction for sqlite3 database), data ingestion (deals with raw data fromcsv), analytics (calculates deterministic stats on dataset), LLM insight generation (abstraction for openai API call), the main (initializes app and serves UI). Please see appendix for detailed designs. 

**AI tool used: what AI tool did you use and for what**
A: I used chatgpt for this project. I used it to 1) generate code. I design the overall systems and interface between modules, and used chatgpt to generate code for each method. 2) learn concepts I'm not familiar with, such as which statistics to calculate on transaction data to find risky patterns.

**Time spent**
A: this took me between 4 - 8 hours of work to finish.


## Appendix: thought process on data processing and database schema
**Data pre-processing criterias:**
- Rows missing value for any column is rejected 
- Timestamp not in valid format is rejected 
- Action not in [BUY, SELL] is rejected 
- Quantity/price can not be parsed as numeric values are rejected
- Quantity/price negative values are rejected 
- All string types are normalized (upper/lower cases)
		
**Dataset schema and design**
- Here I store the clean data in a transaction table with six columns corresponding to the columns of the original csv file 
- Quantity and price are stored as REAL types, the rest as string 
- Date time is converted to isoformat for easy filtering by start and end time
- The rejected rows are stored in a separate table with reason column 
- Meta data is stored in a separate table using the dataset_id (hash of csv file) as key
There is another table that keeps track of datasets, so this database can accommodate different datasets, but for the purpose of this demo this is redundant. 




## Appendix: Data Process and Insight Demo, Designs
I also wrote the following brief design doc in the early stages of designing the system: 

## Goals
- Short rephrase of goals as shown on the OA page.

### Part 1: Data
- Read the data
- Identify and handle potential data quality issues
- Store transactions so that:
  - Lookup by ticker
  - Sorting by time  
  can be done efficiently
- Calculate statistics on the data, including but not limited to:
  - Volume per ticker
  - Net position per ticker
  - Most active `trader_id`


### Part 2: LLM
- Use an LLM to identify patterns, risks, and unusual activities, etc.


### Part 3: Dashboard
- (At least):
  - Load and display CSV data
  - Show calculated statistics
  - Provide a button that shows LLM-generated insights on the data


## Restrictions
- Python is specifically required
- LLM is specifically required (no restriction on which one)
- Data size is on the order of thousands of rows

## Technical Design

### Choice of Tools
- **Overall language**: Python 3  
  - Required and standard for data processing
- **Data library**: Pandas  
  - Industry standard for financial data processing
- **Database**: SQLite  
  - Native Python support, quick iteration for demo purposes
- **LLM**: OpenAI  
  - Industry leading (and has free credit)
- **Visualization**: Streamlit  
  - Lightweight and sufficient for this demo

#### Thought Process for Stack Selection
- Prioritize fast MVP, easy iteration, and lightweight setup
- Prefer running as many components locally as possible

#### In Practice
- Data storage should be handled in a cloud platform:
  - Either company internal infrastructure
  - Or a commercial cloud platform
- Visualization could be implemented as a web app:
  - Developed locally
  - Hosted remotely
- More likely, the company’s internal dashboard system or an external commercial dashboard would be integrated

## System Design

A 5-module system is proposed for the demo. This is a high-level description with a focus on defining clear module boundaries.

### 1. Data Processing
- Parse the CSV file
- Perform required data normalization and cleaning
- Send processed data to the storage module
- Only module that interacts with raw CSV data
- Runs once at application initialization
- Generates statistics about data quality

### 2. Data Storage
- Owns all database interactions
- Provides a data access abstraction for the rest of the application
- Initializes the database
- Runs once at initialization time

### 3. Data Analysis
- Contains all statistical analysis functionality
- Retrieves data via the data storage module

### 4. LLM Insight Query
- Converts statistics from the data analysis module into natural-language insights
- Makes queries to the LLM
- Implements a web API query/response handler
- Abstracted from the rest of the application

### 5. Main and UI
- Serves as the main entry point of the application
- Presents a dashboard GUI
- Displays analytics and LLM-generated insights to the user
