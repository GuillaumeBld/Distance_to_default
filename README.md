# Bank Distance-to-Default (DD) Pipeline

This project implements an automated pipeline for calculating Distance-to-Default (DD) and Probability of Default (PD) for banks using the KMV model. The pipeline fetches market data from Refinitiv, processes financial statements, and outputs the results to an Excel workbook.

## Features

- **Flexible Data Sources**: Fetches price data from multiple sources in this priority:
  1. Local CSV/Excel file (if provided via `--price-file`)
  2. Refinitiv Data Platform (RDP) with automatic fallback to REST API
  3. Yahoo Finance as a final fallback
- **Comprehensive Metrics**: Calculates daily log returns, annualized volatilities, and KMV model metrics
- **Robust Error Handling**: Graceful degradation with clear error messages and fallbacks
- **Detailed Logging**: Verbose logging for debugging and monitoring
- **Excel Export**: Well-formatted output with multiple sheets for analysis

## Prerequisites

- Python 3.8+
- For Refinitiv Data Platform (RDP) access (optional but recommended):
  - RDP credentials (App Key, Client ID, Client Secret)
  - `refinitiv-data` package (installed via requirements.txt)
- For Yahoo Finance fallback (installed automatically):
  - `yfinance` package

### Checking Dependencies

You can check if all required dependencies are installed by running:

```bash
python run_dd_pipeline.py --check-dependencies
```

This will verify that all required packages are installed and meet the minimum version requirements.

### Viewing Configuration

To view the current configuration settings without running the full analysis:

```bash
python run_dd_pipeline.py --show-config
```

This will display information about:
- Version information
- File paths
- Date range
- Model parameters
- Execution settings

### Running Tests

To quickly test the pipeline with a small subset of data:

```bash
python run_dd_pipeline.py --test
```

This will:
- Process only a few tickers (default: 3)
- Use a shorter time period (1 year)
- Save output to a test file (appends `_test` to the output filename)
- Print additional debug information

For more control, you can specify exactly how many tickers to process:

```bash
# Process exactly 5 tickers
python run_dd_pipeline.py --test --max-tickers 5

# Process all tickers but with a limited time period
python run_dd_pipeline.py --test --max-tickers 0
```

This is useful for verifying that the pipeline is working correctly before running a full analysis.

> **Note**: At least one data source (local file, Refinitiv, or Yahoo Finance) must be available.

## Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd bank-dd-pipeline
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Refinitiv credentials:
   ```
   RDP_APP_KEY=your_app_key_here
   RDP_CLIENT_ID=your_client_id_here
   RDP_CLIENT_SECRET=your_client_secret_here
   RISK_FREE_RATE=0.05  # Optional, default is 5%
   ```

## Input Data

Place your bank data in an Excel file named `esg_0426 copy.xlsx` in the project root. The file should contain a sheet named "data" with the following columns:

- `Instrument`: Ticker symbol (e.g., "JPM.N")
- `Total Assets`: Total assets in USD
- `Debt - Total`: Total debt in USD
- `Price to Book Value per Share`: Price to book ratio

## Usage

### Basic Usage

```bash
# Using default settings (tries Refinitiv, falls back to Yahoo Finance)
python run_dd_pipeline.py

# Using a local price file
python run_dd_pipeline.py --price-file path/to/prices.csv

# With custom date range
python run_dd_pipeline.py --start-date 2020-01-01 --end-date 2023-12-31
```

### Advanced Options

```bash
# Run in dry-run mode (don't save output files)
python run_dd_pipeline.py --dry-run

# List all tickers in the input file and exit
python run_dd_pipeline.py --list-tickers

# Check if all dependencies are installed
python run_dd_pipeline.py --check-dependencies

# Show the current configuration
python run_dd_pipeline.py --show-config

# Run a quick test with a small subset of data
python run_dd_pipeline.py --test

# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
python run_dd_pipeline.py --log-level DEBUG

# Enable verbose output (equivalent to --log-level=DEBUG)
python run_dd_pipeline.py -v
# or
python run_dd_pipeline.py --verbose

# Override risk-free rate (default: 0.05 or from .env)
python run_dd_pipeline.py --risk-free-rate 0.03

# Show version and exit
python run_dd_pipeline.py --version

# Show help message with all options
python run_dd_pipeline.py --help
```

## Command-Line Options

### Input/Output Files

- `--input`, `-i` FILE  
  Input Excel file with bank data (default: `esg_0426 copy.xlsx`)

- `--output`, `-o` FILE  
  Output Excel file (default: `dd_results.xlsx`)

- `--price-file`, `-p` FILE  
  Optional CSV/Excel file with historical prices

- `--sheet`, `-s` NAME  
  Sheet name in the input file (default: `data`)

### Date Range

- `--start-date` DATE  
  Start date in YYYY-MM-DD format (default: 5 years before end date)

- `--end-date` DATE  
  End date in YYYY-MM-DD format (default: today)

- `--years` N  
  Number of years of historical data to use (default: 5)

### Model Parameters

- `--risk-free-rate` RATE  
  Risk-free rate as decimal (e.g., 0.05 for 5%). Overrides .env file.

- `--trading-days` DAYS  
  Number of trading days in a year (default: 252)

- `--min-observations` N  
  Minimum number of price observations required (default: 50)

- `--max-tickers` N  
  Maximum number of tickers to process (useful for testing, default: all tickers)

### Execution Control

- `--dry-run`  
  Run the pipeline without saving any output files

- `--list-tickers`  
  List all tickers found in the input file and exit

- `--check-dependencies`  
  Check if all required dependencies are installed and exit

- `--show-config`  
  Show the current configuration and exit

- `--test`  
  Run a quick test with a small subset of data (equivalent to `--max-tickers 3 --years 1`)

- `--log-level` LEVEL  
  Set the logging verbosity level (default: INFO)  
  Choices: DEBUG, INFO, WARNING, ERROR, CRITICAL

- `--verbose`, `-v`  
  Enable verbose output (equivalent to --log-level=DEBUG)

- `--version`, `-V`  
  Show version information and exit

## Data Retrieval Priority

The pipeline attempts to fetch price data in this order:

1. **Local File** (if `--price-file` is provided)
   - Supports both CSV and Excel formats
   - First column should be dates (parsed automatically)
   - Subsequent columns should be ticker symbols with price data

2. **Refinitiv Data Platform**
   - Uses the `refinitiv-data` Python library
   - Falls back to REST API if the library fails
   - Requires valid RDP credentials in `.env`

3. **Yahoo Finance**
   - Used as a final fallback
   - No API key required
   - May have rate limits for large requests

### Output

The script generates an Excel file with these sheets:

1. `KMV_Results`: Main output with all calculated metrics
2. `Daily_Prices`: Historical closing prices
3. `Daily_Returns`: Daily log returns
4. `Summary`: Key statistics and metrics
5. `Parameters`: Input parameters and settings

## Output

The output Excel file contains three sheets:

1. `Daily_Prices`: Historical closing prices for each instrument
2. `Daily_Returns`: Daily log returns for each instrument
3. `KMV_Results`: Calculated metrics for each bank:
   - Market_Equity: Market value of equity
   - Debt_Total: Total debt
   - Equity_Volatility: Annualized equity volatility
   - Asset_Value: Estimated asset value
   - Asset_Volatility: Estimated asset volatility
   - Distance_to_Default: Distance to default (in standard deviations)
   - Probability_of_Default: Probability of default (0-1)

## Logging and Debugging

The script provides detailed logging to help with debugging:

- Console output shows progress and important messages
- Detailed logs are saved to `dd_pipeline.log` (overwritten on each run)
- Debug logs include data shapes, transformation steps, and API interactions
- Log level can be controlled via command line (default: INFO)

### Log Levels

- `DEBUG`: Detailed information, typically of interest only when diagnosing problems
- `INFO`: Confirmation that things are working as expected
- `WARNING`: An indication that something unexpected happened (but the software is still working)
- `ERROR`: Due to a more serious problem, the software has not been able to perform some function
- `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running

### Viewing Logs

Logs are written to both the console and `dd_pipeline.log` in the current directory. The log file is overwritten on each run.

To view logs in real-time:

```bash
# View logs in console at INFO level (default)
python run_dd_pipeline.py

# View detailed debug logs
python run_dd_pipeline.py --log-level DEBUG

# View only errors and above
python run_dd_pipeline.py --log-level ERROR
```

## Error Handling

The script includes comprehensive error handling for:
- Missing or invalid input data
- API authentication and rate limiting
- Numerical issues in the KMV calculations
- File I/O operations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Refinitiv for providing market data APIs
- KMV Corporation for the KMV model methodology
