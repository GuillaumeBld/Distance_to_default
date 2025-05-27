# Bank Risk Analysis Tool - Simple Guide for Non-Coders

## What Does This Tool Do?

This tool helps analyze how risky banks are by looking at their financial health and sustainability practices. It acts as health check-up for banks that tells you:

-  **Safe**: The bank is very stable and unlikely to fail
-  **Caution**: The bank needs monitoring but is generally okay
-  **Risk**: The bank is in trouble and might fail

## What Information Does It Analyze?

The tool looks at three main things:

1. **Financial Health**
   - How much money the bank has (assets)
   - How much the bank owes (debt)
   - The bank's stock price

2. **Sustainability Scores (ESG)**
   - **E**nvironmental: How eco-friendly the bank is
   - **S**ocial: How well the bank treats people and communities
   - **G**overnance: How well the bank is managed

3. **Risk Calculations**
   - **Distance-to-Default**: How far the bank is from failing (higher is better)
   - **Probability of Default**: The chance the bank will fail (0% to 100%)

## How to Use This Tool (Using Docker)

### What You Need First

1. **Docker Desktop** installed on your computer
   - Download from: https://www.docker.com/products/docker-desktop/
   - Install it like any other program

2. **Your Bank Data** in an Excel file
   - The file should have columns for:
     - Bank names/symbols
     - Total assets
     - Total debt
     - Stock prices
     - ESG scores

### Step-by-Step Instructions

#### 1. Get the Tool

Download or copy this entire folder to your computer.

#### 2. Prepare Your Data

Place your Excel file with bank data in the `data` folder. Name it `esg_0426 copy.xlsx` or update the settings to use your filename.

#### 3. Start the Tool

Open a terminal/command prompt in the folder and type:

```bash
docker-compose up
```

This command:
- Downloads everything needed automatically
- Sets up the analysis environment
- Runs the analysis on your data

#### 4. Wait for Results

The tool will:
- Read your Excel file
- Fetch current stock prices (if available)
- Calculate all risk metrics
- Create a new Excel file with results

#### 5. Find Your Results

Look for a file called `dd_results.xlsx` in the folder. This contains:
- All your original data
- Calculated risk scores
- Color-coded risk levels

## Understanding Your Results

### In the Results File

- **Distance_to_Default** column:
  - Above 3 = Very safe bank
  - 1 to 3 = Monitor this bank
  - Below 1 = High risk bank

- **Probability_of_Default** column:
  - Shows percentage chance of failure
  - 0% = No risk
  - 100% = Certain failure

### What the ESG Scores Mean

The tool shows how sustainability practices affect bank stability:
- High ESG scores often mean better long-term stability
- Low scores might indicate future problems

## Troubleshooting

### If Docker Won't Start

1. Make sure Docker Desktop is running (check your system tray)
2. Try restarting Docker Desktop
3. Make sure you have enough disk space

### If the Analysis Fails

1. Check your Excel file has all required columns
2. Make sure bank symbols are correct (e.g., "JPM" for JP Morgan)
3. Check the `dd_pipeline.log` file for error messages

### If You Can't Find Results

The results file `dd_results.xlsx` will be in the same folder as this README.


## What Happens Behind the Scenes?

The tool:
1. Reads your bank data from Excel
2. Fetches current stock prices from financial databases
3. Uses advanced mathematical models (KMV model) to calculate risk
4. Combines financial risk with ESG scores
5. Outputs everything in an easy-to-read Excel file

