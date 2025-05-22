#!/usr/bin/env python3

import os
import sys
import importlib
import argparse
import logging
import platform
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import requests
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Version information
__version__ = "1.1.0"  # Major.Minor.Patch format

# Version information string
VERSION_INFO = f"""
Distance-to-Default (DD) Pipeline v{__version__}
Python {sys.version.split()[0]} on {platform.platform()}
Pandas {pd.__version__}, NumPy {np.__version__}, SciPy {scipy.__version__}
""".strip()

def setup_logging(log_level_str=None):
    """Configure logging with level from environment variable or parameter.
    
    Args:
        log_level_str: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                      If None, uses LOG_LEVEL from environment or defaults to INFO.
                      
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get log level from parameter or environment variable
    log_level_str = log_level_str or os.getenv('LOG_LEVEL', 'INFO')
    log_level_str = log_level_str.upper()
    
    # Map string level to logging constant
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        print(f"Invalid log level '{log_level_str}'. Defaulting to INFO.", file=sys.stderr)
        log_level = logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = logging.FileHandler('dd_pipeline.log', mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Set log level for external libraries (reduce verbosity)
    for lib in ['urllib3', 'requests', 'refinitiv', 'yfinance', 'matplotlib']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured with level: {logging.getLevelName(log_level)}")
    
    return logger

# Constants
TRADING_DAYS = 252  # Number of trading days in a year

def load_environment(logger=None):
    """Load and validate environment variables.
    
    Args:
        logger: Optional logger instance for logging messages
        
    Returns:
        float: Risk-free rate to use in calculations
        
    Raises:
        ValueError: If required environment variables are missing or invalid
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    load_dotenv()
    
    # Log environment variables (masking sensitive values)
    env_vars = {
        'RDP_APP_KEY': os.getenv('RDP_APP_KEY', None),
        'RDP_CLIENT_ID': os.getenv('RDP_CLIENT_ID', None),
        'RDP_CLIENT_SECRET': '***' if os.getenv('RDP_CLIENT_SECRET') else None,
        'REFINITIV_API_TOKEN': '***' if os.getenv('REFINITIV_API_TOKEN') else None,
        'RISK_FREE_RATE': os.getenv('RISK_FREE_RATE', '0.05')
    }
    
    logger.debug("Environment variables loaded")
    
    # Validate required environment variables
    missing_vars = [var for var in ['RDP_APP_KEY', 'RDP_CLIENT_ID', 'RDP_CLIENT_SECRET'] 
                   if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.warning("Refinitiv data access will fail without these variables")
    
    # Parse risk-free rate
    try:
        risk_free_rate = float(os.getenv('RISK_FREE_RATE', '0.05'))
        logger.info(f"Using risk-free rate: {risk_free_rate:.2%}")
        return risk_free_rate
    except ValueError as e:
        logger.error(f"Invalid RISK_FREE_RATE: {os.getenv('RISK_FREE_RATE')}")
        raise ValueError("RISK_FREE_RATE must be a valid number") from e

def load_and_prepare_data(input_file, sheet_name='data', start_date=None, end_date=None, logger=None):
    """Load and prepare bank data from Excel file.
    
    Args:
        input_file: Path to input Excel file
        sheet_name: Name of the sheet to read
        start_date: Optional start date (inclusive) in 'YYYY-MM-DD' format
        end_date: Optional end date (inclusive) in 'YYYY-MM-DD' format
        logger: Optional logger instance for logging messages
        
    Returns:
        DataFrame with prepared bank data
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Loading data from {input_file}")
    
    # Read the Excel file
    bank_data = pd.read_excel(input_file)
    
    # Standardize column names
    bank_data.columns = [col.strip() for col in bank_data.columns]
    
    # Ensure required columns exist
    required_columns = ['Instrument', 'Date', 'Price to Book Value per Share', 
                       'Total Assets', 'Debt - Total']
    missing_columns = [col for col in required_columns if col not in bank_data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Convert date column to datetime if it's not already
    bank_data['Date'] = pd.to_datetime(bank_data['Date'])
    
    # Filter by date range if specified
    if start_date:
        start_date = pd.to_datetime(start_date)
        bank_data = bank_data[bank_data['Date'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        bank_data = bank_data[bank_data['Date'] <= end_date]
    
    if bank_data.empty:
        raise ValueError("No data available for the specified date range")
    
    # Convert numeric columns and handle missing values
    numeric_cols = ['Price to Book Value per Share', 'Total Assets', 'Debt - Total']
    bank_data[numeric_cols] = bank_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Calculate Market Cap = Price-to-Book × (Total_Assets – Total_Debt)
    bank_data['Market Cap'] = bank_data['Price to Book Value per Share'] * \
                             (bank_data['Total Assets'] - bank_data['Debt - Total'])
    
    # Sort by date for each instrument
    bank_data = bank_data.sort_values(['Instrument', 'Date'])
    
    # Log summary
    date_range = f"from {bank_data['Date'].min().date()} to {bank_data['Date'].max().date()}"
    logger.info(f"Loaded {len(bank_data)} bank records {date_range} for "
               f"{bank_data['Instrument'].nunique()} unique instruments")
    
    return bank_data

def load_price_file(path):
    """Load price data from a local file (CSV or Excel).
    
    Args:
        path: Path to the price data file (CSV or Excel)
        
    Returns:
        DataFrame with dates as index and tickers as columns
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is empty or has invalid format
    """
    logger.info(f"Loading price data from local file: {path}")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Price file not found: {path}")
    
    try:
        # Read the file based on extension
        if path.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(path, index_col=0, parse_dates=True)
        elif path.lower().endswith('.csv'):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        # Basic validation
        if df.empty:
            raise ValueError("Price file is empty")
            
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort index
        df = df.sort_index()
        
        # Log success
        logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} tickers from {path}")
        logger.debug(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading price file {path}: {str(e)}")
        raise

def fetch_prices_yahoo(tickers, start_date, end_date, logger=None):
    """Fetch historical prices using Yahoo Finance API.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        logger: Optional logger instance for logging messages
        
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    import yfinance as yf
    
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Fetching Yahoo prices for {len(tickers)} tickers from {start_date} to {end_date}")
    
    try:
        # Download data with progress disabled for cleaner logs
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker'
        )
        
        # Extract Close prices and format
        if len(tickers) == 1:
            df = raw[['Close']].rename(columns={'Close': tickers[0]})
        else:
            df = raw.xs('Close', level=1, axis=1)
        
        logger.debug(f"Yahoo price data shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Yahoo Finance fetch failed: {e}")
        raise

def _fetch_prices_rdp_library(tickers, start_date, end_date, logger=None):
    """Fetch historical prices using Refinitiv Data Platform library.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        logger: Optional logger instance for logging messages
        
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        import refinitiv.data as rd
        from refinitiv.data import Session
        
        # Initialize session if not already done
        if not Session.get_default_session().is_initialized:
            rd.open_session()
            
        # Fetch data for each ticker
        dfs = []
        for ticker in tickers:
            try:
                # Try with .O first (common suffix for stocks)
                try:
                    data = rd.get_data(
                        universe=[f"{ticker}.O"],
                        fields=['TR.PriceClose'],
                        parameters={
                            'SDate': start_date,
                            'EDate': end_date,
                            'Frq': 'D'
                        }
                    )
                    if not data.empty:
                        data['Instrument'] = ticker
                        dfs.append(data)
                        continue
                except Exception as e:
                    logger.debug(f"Ticker {ticker}.O not found, trying without suffix: {str(e)}")
                    
                # Try without .O if that fails
                try:
                    data = rd.get_data(
                        universe=[ticker],
                        fields=['TR.PriceClose'],
                        parameters={
                            'SDate': start_date,
                            'EDate': end_date,
                            'Frq': 'D'
                        }
                    )
                    if not data.empty:
                        data['Instrument'] = ticker
                        dfs.append(data)
                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker}: {str(e)}")
                    
        if not dfs:
            raise ValueError("No data could be fetched for any ticker")
            
        # Combine and pivot
        df = pd.concat(dfs)
        df = df.pivot(index='Date', columns='Instrument', values='Price Close')
        logger.info(f"Successfully fetched data for {len(df.columns)} tickers")
        return df
        
    except Exception as e:
        logger.error(f"Refinitiv library fetch failed: {str(e)}")
        raise

def _fetch_prices_rdp_rest(tickers, start_date, end_date, token, logger=None):
    """Fallback to Refinitiv REST API if library fails.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        token: API token for authentication
        logger: Optional logger instance for logging messages
        
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if not token:
        raise ValueError("No token provided for REST API fallback")
        
    base_url = "https://api.refinitiv.com/data/historical-pricing/v1/"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    try:
        # Make API request
        response = requests.post(
            f"{base_url}views/events/"
            f"{','.join(tickers)}/prices.Closing"
            f"?start={start_date}&end={end_date}",
            headers=headers
        )
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        df = pd.DataFrame(data['data'])
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.pivot(index='date', columns='ric', values='value')
            logger.debug(f"Refinitiv REST data shape: {df.shape}")
        
        return df
    except Exception as e:
        logger.warning(f"Refinitiv REST API fetch failed: {str(e)}")
        raise

def fetch_prices_refinitiv(tickers, start_date, end_date, token=None, logger=None):
    """Fetch historical prices from Refinitiv with fallback to REST API.
    
    Args:
        tickers: List of instrument symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        token: Optional API token for REST fallback
        logger: Optional logger instance for logging messages
        
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Fetching Refinitiv prices for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Try Refinitiv library first
    try:
        return _fetch_prices_rdp_library(tickers, start_date, end_date, logger=logger)
    except Exception as e:
        logger.warning(f"Falling back to Refinitiv REST API: {str(e)}")
        return _fetch_prices_rdp_rest(tickers, start_date, end_date, token, logger=logger)

def calculate_returns(prices_df, min_observations=50):
    """Calculate daily log returns from price data.
    
    Args:
        prices_df: DataFrame with Date index and ticker columns
        min_observations: Minimum number of observations required for a valid series
        
    Returns:
        DataFrame of log returns with NaN for series with insufficient data
    """
    logger.info("Calculating daily log returns")
    
    # Forward fill any missing values within each instrument's time series
    prices_ffill = prices_df.ffill()
    
    # Calculate log returns (ln(Pt/Pt-1))
    returns_df = np.log(prices_ffill / prices_ffill.shift(1))
    
    # Replace infinite values with NaN
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    
    # Log shapes and sample data for debugging
    logger.debug(f"Prices shape: {prices_df.shape}, Returns shape: {returns_df.shape}")
    logger.debug(f"First 3 rows of returns:\n{returns_df.head(3).to_string()}")
    
    return returns_df

def calculate_volatility(returns_df, min_observations=50):
    """Calculate annualized volatility from daily returns.
    
    Args:
        returns_df: DataFrame of daily returns
        min_observations: Minimum number of observations required
        
    Returns:
        Series of annualized volatilities with NaN for insufficient data
    """
    logger.info("Calculating annualized volatilities")
    
    # Count valid observations per column
    valid_obs = returns_df.count()
    insufficient_data = valid_obs < min_observations
    
    # Log tickers with insufficient data
    if insufficient_data.any():
        invalid_tickers = valid_obs[insufficient_data].index.tolist()
        logger.warning(f"Insufficient data (<{min_observations} obs) for tickers: {', '.join(invalid_tickers)}")
    
    # Calculate standard deviation and annualize only for valid series
    daily_vol = returns_df.std()
    daily_vol[insufficient_data] = np.nan  # Set NaN for insufficient data
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)
    
    return annual_vol

def kmv_equation(vars_, E, D, r, sigma_E, T=1.0):
    """System of equations for KMV model.
    
    Args:
        vars_: Tuple of (V, sigma_A) - asset value and volatility
        E: Market value of equity
        D: Face value of debt
        r: Risk-free rate (annualized)
        sigma_E: Equity volatility (annualized)
        T: Time horizon in years (default: 1 year)
        
    Returns:
        List of equation residuals
    """
    V, sigma_A = vars_
    sqrt_T = np.sqrt(T)
    
    try:
        # Calculate d1 with time horizon
        d1 = (np.log(V/(D + 1e-10)) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * sqrt_T + 1e-10)
        d2 = d1 - sigma_A * sqrt_T
        
        # Black-Scholes-Merton equations
        E_calc = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
        
        # Volatility equation using cdf(d1)
        sigma_E_calc = (V / (E_calc + 1e-10)) * norm.cdf(d1) * sigma_A
        
        return [E_calc - E, sigma_E_calc - sigma_E]
    except Exception as e:
        logger.debug(f"Error in kmv_equation: {str(e)}")
        return [np.nan, np.nan]

def calculate_kmv_metrics(E, D, r, sigma_E, T=1.0):
    """Calculate KMV metrics using numerical optimization.
    
    Args:
        E: Market value of equity
        D: Face value of debt
        r: Risk-free rate (annualized)
        sigma_E: Equity volatility (annualized)
        T: Time horizon in years (default: 1 year)
        
    Returns:
        Tuple of (V_asset, sigma_A, DD, PD, converged)
    """
    try:
        # Initial guess for asset value
        V0 = E + D  # Initial guess: asset value = equity + debt
        
        # Initial guess for asset volatility with floor at 5%
        sigma_A0 = sigma_E * (E / (E + D + 1e-10))
        sigma_A0 = max(sigma_A0, 0.05)  # Enforce minimum 5% volatility
        
        # Solve the system of equations with full output
        solution, infodict, ier, msg = fsolve(
            kmv_equation,
            [V0, sigma_A0],
            args=(E, D, r, sigma_E, T),
            xtol=1e-8,
            maxfev=1000,
            full_output=1  # Return full output for convergence check
        )
        
        # Check if solution converged
        if ier != 1:
            logger.warning(f"Solver did not converge: {msg}")
            return np.nan, np.nan, np.nan, np.nan, False
        
        V_asset, sigma_A = solution
        
        # Ensure reasonable values
        if V_asset <= 0 or sigma_A <= 0 or sigma_A > 5.0:  # 500% max vol
            logger.warning(f"Unreasonable solution: V={V_asset:.2f}, σA={sigma_A:.4f}")
            return np.nan, np.nan, np.nan, np.nan, False
        
        # Calculate distance to default and probability of default
        sqrt_T = np.sqrt(T)
        d1 = (np.log(V_asset/(D + 1e-10)) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * sqrt_T + 1e-10)
        DD = d1 - sigma_A * sqrt_T
        PD = norm.cdf(-DD)
        
        return V_asset, sigma_A, DD, PD, True
        
    except Exception as e:
        logger.warning(f"Error in KMV calculation: {str(e)}", exc_info=True)
        return np.nan, np.nan, np.nan, np.nan, False

def process_bank_data(bank_data, returns_df, risk_free_rate, T=1.0, min_returns=50, logger=None):
    """Process bank data to calculate KMV metrics for each bank/date combination.
    
    Args:
        bank_data: DataFrame with bank information including dates
        returns_df: DataFrame with daily returns (index=date, columns=instruments)
        risk_free_rate: Annual risk-free rate (float)
        T: Time horizon in years (default: 1.0)
        min_returns: Minimum number of returns required for volatility calculation
        logger: Optional logger instance for logging messages
        
    Returns:
        DataFrame with KMV metrics for each bank/date combination
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Processing KMV metrics with T={T} years")
    
    # Prepare results container
    results = []
    
    # Group by instrument and date
    grouped = bank_data.groupby(['Instrument', 'Date'])
    
    for (ticker, date), group in grouped:
        try:
            # Get the most recent data for this bank/date
            bank_row = group.iloc[0]
            
            # Extract required values
            E = bank_row['Market Cap']  # Market value of equity
            D = bank_row['Debt - Total']  # Total debt
            
            # Skip if missing required values
            if pd.isna(E) or pd.isna(D) or E <= 0 or D <= 0:
                logger.warning(f"Skipping {ticker} on {date.date()}: Invalid E={E:.2f}, D={D:.2f}")
                continue

            # Calculate equity volatility (annualized)
            if ticker not in returns_df.columns:
                logger.warning(f"No returns data for {ticker}")
                continue
                
            returns = returns_df[ticker].dropna()
            if len(returns) < min_returns:
                logger.warning(f"Insufficient returns data for {ticker} on {date.date()} "
                              f"(have {len(returns)}, need {min_returns})")
                continue
                
            sigma_E = returns.std() * np.sqrt(252)  # Annualize
            
            if sigma_E <= 0 or not np.isfinite(sigma_E):
                logger.warning(f"Invalid sigma_E={sigma_E:.6f} for {ticker} on {date.date()}")
                continue
                
            # Calculate KMV metrics
            V_A, sigma_A, DD, PD, converged = calculate_kmv_metrics(E, D, risk_free_rate, sigma_E, T)
            
            if not converged or not all(np.isfinite([V_A, sigma_A, DD, PD])):
                logger.warning(f"KMV solver did not converge for {ticker} on {date.date()}")
                continue
                
            # Collect results
            result = {
                'Date': date,
                'Instrument': ticker,
                'Market_Value_Equity': E,
                'Total_Debt': D,
                'Equity_Volatility': sigma_E,
                'Asset_Value': V_A,
                'Asset_Volatility': sigma_A,
                'Distance_to_Default': DD,
                'Probability_of_Default': PD,
                'Time_Horizon_Years': T,
                'Risk_Free_Rate': risk_free_rate,
                'Converged': converged
            }
            
            # Add any additional columns from the original data
            for col in bank_row.index:
                if col not in result and col not in ['Date', 'Instrument']:
                    result[col] = bank_row[col]
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {ticker} on {date.date()}: {str(e)}", exc_info=True)
    
    if not results:
        logger.error("No valid KMV metrics could be calculated")
        return pd.DataFrame()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    cols = ['Date', 'Instrument', 'Market_Value_Equity', 'Total_Debt', 
            'Asset_Value', 'Asset_Volatility', 'Equity_Volatility',
            'Distance_to_Default', 'Probability_of_Default', 'Time_Horizon_Years',
            'Risk_Free_Rate', 'Converged']
    
    # Add any remaining columns
    extra_cols = [c for c in results_df.columns if c not in cols]
    results_df = results_df[cols + extra_cols]
    
    logger.info(f"Processed KMV metrics for {len(results_df)} bank/date combinations")
    return results_df

def export_results(prices_df, returns_df, kmv_results, output_file, dry_run=False):
    """Export results to Excel file with proper formatting.
    
    Args:
        prices_df: DataFrame with daily prices
        returns_df: DataFrame with daily returns
        kmv_results: DataFrame with KMV metrics
        output_file: Path to output Excel file
        dry_run: If True, don't actually write any files
        
    Returns:
        bool: True if export was successful or if dry_run is True
    """
    logger.info(f"Exporting results to {output_file}")
    
    if dry_run:
        logger.info("Dry run: Skipping file write")
        return True
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Create a workbook and add formats
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            float_fmt = workbook.add_format({'num_format': '#,##0.00'})
            pct_fmt = workbook.add_format({'num_format': '0.00%'})
            int_fmt = workbook.add_format({'num_format': '#,##0'})
            date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            
            # Format for probability of default (color-coded)
            pd_fmt_high = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'num_format': '0.00%'})
            pd_fmt_medium = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700', 'num_format': '0.00%'})
            pd_fmt_low = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'num_format': '0.00%'})
            
            # Export KMV Results
            kmv_results.to_excel(writer, sheet_name='KMV_Results', index=False)
            worksheet = writer.sheets['KMV_Results']
            
            # Format header
            for col_num, value in enumerate(kmv_results.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Format columns
            for col_num, col_name in enumerate(kmv_results.columns):
                # Set column width
                max_length = max(kmv_results[col_name].astype(str).map(len).max(), len(col_name)) + 2
                worksheet.set_column(col_num, col_num, min(max_length, 30))
                
                # Apply number formatting
                if 'Date' in col_name:
                    worksheet.set_column(col_num, col_num, 12, date_fmt)
                elif 'Probability' in col_name or 'Volatility' in col_name or 'Rate' in col_name:
                    worksheet.set_column(col_num, col_num, 15, pct_fmt)
                elif col_name in ['Market_Value_Equity', 'Face_Value_Debt', 'Asset_Value']:
                    worksheet.set_column(col_num, col_num, 18, float_fmt)
                elif col_name == 'Distance_to_Default':
                    worksheet.set_column(col_num, col_num, 15, float_fmt)
            
            # Apply conditional formatting to Probability_of_Default
            if 'Probability_of_Default' in kmv_results.columns:
                pd_col = kmv_results.columns.get_loc('Probability_of_Default')
                worksheet.conditional_format(
                    1, pd_col, len(kmv_results), pd_col,
                    {'type': 'cell', 'criteria': '>=', 'value': 0.1, 'format': pd_fmt_high}
                )
                worksheet.conditional_format(
                    1, pd_col, len(kmv_results), pd_col,
                    {'type': 'cell', 'criteria': 'between', 'minimum': 0.05, 'maximum': 0.1, 'format': pd_fmt_medium}
                )
                worksheet.conditional_format(
                    1, pd_col, len(kmv_results), pd_col,
                    {'type': 'cell', 'criteria': '<', 'value': 0.05, 'format': pd_fmt_low}
                )
            
            # Add a summary sheet
            if not kmv_results.empty:
                summary_data = {
                    'Metric': [
                        'Number of Bank/Date Pairs', 
                        'Average Distance to Default', 
                        'Median Distance to Default', 
                        'Default Probability (Avg)',
                        'Default Probability (Median)',
                        'Average Asset Volatility',
                        'Average Equity Volatility',
                        'Average Asset Value',
                        'Average Market Cap'
                    ],
                    'Value': [
                        len(kmv_results),
                        kmv_results['Distance_to_Default'].mean(),
                        kmv_results['Distance_to_Default'].median(),
                        kmv_results['Probability_of_Default'].mean(),
                        kmv_results['Probability_of_Default'].median(),
                        kmv_results['Asset_Volatility'].mean() if 'Asset_Volatility' in kmv_results.columns else 0,
                        kmv_results['Equity_Volatility'].mean() if 'Equity_Volatility' in kmv_results.columns else 0,
                        kmv_results['Asset_Value'].mean() if 'Asset_Value' in kmv_results.columns else 0,
                        kmv_results['Market_Value_Equity'].mean() if 'Market_Value_Equity' in kmv_results.columns else 0
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                worksheet = writer.sheets['Summary']
                for col_num, value in enumerate(summary_df.columns):
                    worksheet.write(0, col_num, value, header_format)
                
                # Format numbers in summary
                for row in range(1, len(summary_df) + 1):
                    if row == 1:  # Count of bank/date pairs
                        worksheet.write(row, 1, int(summary_df.iloc[row-1, 1]), int_fmt)
                    elif row in [2, 3]:  # DD values
                        worksheet.write(row, 1, summary_df.iloc[row-1, 1], float_fmt)
                    elif row in [4, 5, 6, 7]:  # PD and Volatility values
                        worksheet.write(row, 1, summary_df.iloc[row-1, 1], pct_fmt)
                    else:  # Monetary values
                        worksheet.write(row, 1, summary_df.iloc[row-1, 1], float_fmt)
                
                worksheet.set_column(0, 0, 30)
                worksheet.set_column(1, 1, 20)
            
            # Export prices if provided
            if prices_df is not None and not prices_df.empty:
                prices_df.to_excel(writer, sheet_name='Daily_Prices')
                worksheet = writer.sheets['Daily_Prices']
                
                # Format header
                for col_num, value in enumerate(prices_df.columns):
                    worksheet.write(0, col_num + 1, value, header_format)
                
                # Format date column and price columns
                worksheet.set_column('A:A', 12, date_fmt)
                for col in range(1, len(prices_df.columns) + 1):
                    worksheet.set_column(col, col, 12, float_fmt)
            
            # Export returns if provided
            if returns_df is not None and not returns_df.empty:
                returns_df.to_excel(writer, sheet_name='Daily_Returns')
                worksheet = writer.sheets['Daily_Returns']
                
                # Format header
                for col_num, value in enumerate(returns_df.columns):
                    worksheet.write(0, col_num + 1, value, header_format)
                
                # Format date column and return columns
                worksheet.set_column('A:A', 12, date_fmt)
                for col in range(1, len(returns_df.columns) + 1):
                    worksheet.set_column(col, col, 12, pct_fmt)
            
            # Add a sheet with input parameters
            params_data = {
                'Parameter': [
                    'Generated At',
                    'Risk-Free Rate', 
                    'Time Horizon (Years)', 
                    'Start Date', 
                    'End Date', 
                    'Min Returns for Volatility',
                    'Output File'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    kmv_results['Risk_Free_Rate'].iloc[0] if not kmv_results.empty and 'Risk_Free_Rate' in kmv_results.columns else 'N/A',
                    kmv_results['Time_Horizon_Years'].iloc[0] if not kmv_results.empty and 'Time_Horizon_Years' in kmv_results.columns else 'N/A',
                    kmv_results['Date'].min().strftime('%Y-%m-%d') if not kmv_results.empty and 'Date' in kmv_results.columns else 'N/A',
                    kmv_results['Date'].max().strftime('%Y-%m-%d') if not kmv_results.empty and 'Date' in kmv_results.columns else 'N/A',
                    returns_df.count().min() if returns_df is not None and not returns_df.empty else 'N/A',
                    os.path.basename(output_file)
                ]
            }
            
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            # Format parameters sheet
            worksheet = writer.sheets['Parameters']
            for col_num, value in enumerate(params_df.columns):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet.set_column(0, 0, 25)
            worksheet.set_column(1, 1, 30)
            
            # Format the generated at timestamp
            worksheet.write_datetime(1, 1, datetime.now(), workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'}))
        
        logger.info(f"Results successfully exported to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}", exc_info=True)
        return False

def clean_temp_files():
    """Remove temporary files and cached data.
    
    Returns:
        int: Number of files removed
    """
    import glob
    import shutil
    
    temp_files = [
        'dd_pipeline.log',  # Log file
        'refinitiv_token.json',  # Cached Refinitiv token
        '*.pkl',  # Any pickle files
        '*.tmp',  # Temporary files
        '*.cache',  # Cache files
        '*.log.*',  # Rotated log files
    ]
    
    removed = 0
    for pattern in temp_files:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed file: {file_path}")
                    removed += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.debug(f"Removed directory: {file_path}")
                    removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")
    
    return removed

def check_requirements(logger=None):
    """Check if Python version and all required packages are installed with correct versions.
    
    Args:
        logger: Optional logger instance for logging warnings
        
    Raises:
        RuntimeError: If Python version is too old
        ImportError: If required packages are missing or have incorrect versions
    """
    # Check Python version first
    if sys.version_info < (3, 8):
        raise RuntimeError(
            f"Python 3.8 or higher is required. You are using Python {platform.python_version()}."
        )
    
    # Required packages with minimum versions
    required_packages = {
        'pandas': '1.3.0',
        'numpy': '1.21.0',
        'scipy': '1.7.0',
        'requests': '2.26.0',
        'openpyxl': '3.0.9',  # For Excel I/O
        'python-dotenv': '0.19.0',
        'yfinance': '0.1.70',
        'xlsxwriter': '3.0.2',
        'packaging': '20.0',  # For version parsing
        # Refinitiv packages are optional but recommended
    }
    
    # Optional packages (will warn but not fail if missing/old)
    optional_packages = {
        'refinitiv_data': '1.0.0',
    }
    
    missing_packages = []
    wrong_version = []
    optional_issues = []
    
    def parse_version(version_str):
        """Parse version string into a comparable tuple."""
        from packaging import version
        try:
            return version.parse(version_str)
        except version.InvalidVersion:
            # If version string is not PEP 440 compliant, try to parse it
            return version.parse('0.0.0')
    
    # Check required packages
    for package, min_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            current_version = getattr(module, '__version__', '0.0.0')
            
            # Compare versions using packaging library
            if parse_version(current_version) < parse_version(min_version):
                warning_msg = f"{package} {current_version} is installed but requires {min_version}+"
                if logger:
                    logger.warning(warning_msg)
                else:
                    print(f"WARNING: {warning_msg}")
                wrong_version.append(f"{package} {current_version} (requires {min_version}+)")
                
        except ImportError:
            missing_packages.append(package)
    
    # Check optional packages
    for package, min_version in optional_packages.items():
        try:
            module = importlib.import_module(package.replace('-', '_'))
            current_version = getattr(module, '__version__', '0.0.0')
            
            if parse_version(current_version) < parse_version(min_version):
                optional_issues.append(f"{package} {current_version} (optional, but {min_version} recommended)")
                
        except ImportError:
            optional_issues.append(f"{package} not installed (optional, but recommended)")
    
    # Build error message if needed
    error_msg = []
    
    if missing_packages or wrong_version:
        if missing_packages:
            error_msg.append(f"Missing required packages: {', '.join(missing_packages)}")
        if wrong_version:
            error_msg.append(f"Incorrect package versions: {', '.join(wrong_version)}")
        
        error_msg.append(
            "\nPlease install the required packages using:"
            "\n  pip install -r requirements.txt"
            "\n\nOr install a specific version manually, for example:"
            "\n  pip install pandas>=1.3.0 numpy>=1.21.0 ..."
        )
    # Create a custom formatter that preserves formatting in the help text
    class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings:
                metavar, = self._metavar_formatter(action, action.dest)(1)
                return metavar
            else:
                parts = []
                # If the Optional doesn't take a value, format is:
                #    -s, --long
                if action.nargs == 0:
                    parts.extend(action.option_strings)
                # If the Optional takes a value, format is:
                #    -s ARGS, --long ARGS
                else:
                    default = action.default if action.default is not argparse.SUPPRESS else None
                    args_string = self._format_args(action, action.dest.upper())
                    for option_string in action.option_strings:
                        parts.append(f"{option_string} {args_string}")
                        if default and '%(default)' not in action.help:
                            action.help += f" (default: %(default)s)"
                return ', '.join(parts)
    
    # Create the parser with custom formatter
    parser = argparse.ArgumentParser(
        description='''Calculate Distance-to-Default (DD) and Probability of Default (PD) for banks.
        
This tool implements the KMV model to estimate the distance-to-default and probability of default
for financial institutions using market data and financial statement information.
        
Examples:
  # Basic usage with default settings
  python %(prog)s
  
  # Use a local price file
  python %(prog)s --price-file prices.csv
  
  # Specify custom date range
  python %(prog)s --start-date 2020-01-01 --end-date 2023-12-31
  
  # Run in test mode with limited data
  python %(prog)s --test
  
  # Get help
  python %(prog)s --help
''',
        formatter_class=CustomHelpFormatter,
        add_help=False,  # We'll add help manually to control its position
        epilog='''
Examples:
  # Basic usage with default settings
  python %(prog)s
  
  # Specify input and output files
  python %(prog)s --input banks.xlsx --output results.xlsx
  
  # Use a local price file and set date range
  python %(prog)s --price-file prices.csv --start-date 2020-01-01 --end-date 2023-12-31
  
  # Run in dry-run mode with debug logging
  python %(prog)s --dry-run --log-level DEBUG
''')
    
    # Add help argument manually to control its position
    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.'
    )
    
    # Version
    parser.add_argument('--version', '-V', action='version',
                      version=f'%(prog)s {__version__}\n{VERSION_INFO}',
                      help='Show version information and exit')
    parser.add_argument('--clean', action='store_true',
                      help='Remove temporary files and cached data before running')
    
    # Input/Output arguments
    input_group = parser.add_argument_group('Input/Output Options')
    input_group.add_argument('--input', type=str, default='esg_0426 copy.xlsx',
                           help='Input Excel file with bank data')
    input_group.add_argument('--sheet', type=str, default='data',
                           help='Sheet name in the input Excel file')
    input_group.add_argument('--output', type=str, default='dd_results.xlsx',
                           help='Output Excel file for results')
    input_group.add_argument('--price-file', type=str,
                           help='Optional CSV/Excel file with historical prices (first column: dates, subsequent columns: ticker symbols)')
    
    # Date range arguments
    date_group = parser.add_argument_group('Date Range Options')
    date_group.add_argument('--start-date', type=str,
                          help='Start date (YYYY-MM-DD). If not provided, defaults to 3 years before end date')
    date_group.add_argument('--end-date', type=str,
                          help='End date (YYYY-MM-DD). If not provided, defaults to today')
    date_group.add_argument('--years', type=int, default=3,
                          help='Number of years of historical data to fetch (used if start-date not provided)')
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--risk-free-rate', type=float,
                           help='Risk-free rate (as decimal, e.g., 0.05 for 5%%. Overrides .env)')
    model_group.add_argument('--trading-days', type=int, default=252,
                           help='Number of trading days in a year')
    model_group.add_argument('--min-observations', type=int, default=50,
                           help='Minimum number of price observations required for calculations')
    model_group.add_argument('--max-tickers', type=int, default=None,
                           help='Maximum number of tickers to process (useful for testing)')
    
    # Execution control
    exec_group = parser.add_argument_group('Execution Control')
    exec_group.add_argument('--dry-run', action='store_true',
                          help='Run the pipeline without saving any output files')
    exec_group.add_argument('--list-tickers', action='store_true',
                          help='List all tickers found in the input file and exit')
    exec_group.add_argument('--check-dependencies', action='store_true',
                          help='Check if all required dependencies are installed and exit')
    exec_group.add_argument('--show-config', action='store_true',
                          help='Show the current configuration and exit')
    exec_group.add_argument('--test', action='store_true',
                          help='Run a quick test with a small subset of data')
    exec_group.add_argument('--log-level', type=str.upper, default='INFO',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                          help='Set the logging verbosity level')
    exec_group.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose output (equivalent to --log-level=DEBUG)')
    
    return parser.parse_args()

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    
    # Create the parser with RawTextHelpFormatter to handle % signs in help text
    parser = argparse.ArgumentParser(
        description='Calculate Distance-to-Default (DD) and Probability of Default (PD) for banks.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Input/Output options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument(
        '--input', '-i',
        default='esg_0426.xlsx',
        help='Input Excel file with bank data'
    )
    io_group.add_argument(
        '--output', '-o',
        default='dd_results.xlsx',
        help='Output Excel file for results'
    )
    io_group.add_argument(
        '--price-file', '-p',
        help='Optional CSV/Excel file with historical prices'
    )
    io_group.add_argument(
        '--sheet', '-s',
        default='data',
        help='Sheet name in the input file'
    )
    
    # Date range options
    date_group = parser.add_argument_group('Date Range Options')
    date_group.add_argument(
        '--start-date',
        help='Start date in YYYY-MM-DD format (default: 5 years before end date)'
    )
    date_group.add_argument(
        '--end-date',
        help='End date in YYYY-MM-DD format (default: today)'
    )
    date_group.add_argument(
        '--years',
        type=int,
        default=5,
        help='Number of years of historical data to use'
    )
    
    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument(
        '--risk-free-rate',
        type=float,
        help='Risk-free rate (as decimal, e.g., 0.05 for 5%%. Overrides .env)'
    )
    model_group.add_argument(
        '--trading-days',
        type=int,
        default=252,
        help='Number of trading days in a year'
    )
    model_group.add_argument(
        '--min-observations',
        type=int,
        default=50,
        help='Minimum number of price observations required'
    )
    model_group.add_argument(
        '--max-tickers',
        type=int,
        help='Maximum number of tickers to process (useful for testing)'
    )
    
    # Execution control
    exec_group = parser.add_argument_group('Execution Control')
    exec_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the pipeline without saving any output files'
    )
    exec_group.add_argument(
        '--list-tickers',
        action='store_true',
        help='List all tickers found in the input file and exit'
    )
    exec_group.add_argument(
        '--check-dependencies',
        action='store_true',
        help='Check if all required dependencies are installed and exit'
    )
    exec_group.add_argument(
        '--show-config',
        action='store_true',
        help='Show the current configuration and exit'
    )
    exec_group.add_argument(
        '--test',
        action='store_true',
        help='Run a quick test with a small subset of data'
    )
    exec_group.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging verbosity level'
    )
    exec_group.add_argument(
        '--verbose', '-v',
        action='store_const',
        const='DEBUG',
        dest='log_level',
        help='Enable verbose output (equivalent to --log-level=DEBUG)'
    )
    # Create a simple version string
    version_str = f'%(prog)s {__version__}'
    exec_group.add_argument('--version', '-V', action='version',
                      version=version_str,
                      help='Show version information and exit')
    exec_group.add_argument('--clean', action='store_true',
                      help='Remove temporary files and cached data before running')
    
    return parser.parse_args()

def main():
    """Main function to run the DD pipeline."""
    # Set up logging with default level first
    logger = setup_logging('INFO')
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Update logging level based on arguments
        if hasattr(args, 'verbose') and args.verbose:
            logger = setup_logging('DEBUG')
            args.log_level = 'DEBUG'
        else:
            logger = setup_logging(args.log_level)
            
        # Log startup information
        logger.info("=" * 80)
        logger.info(f"Starting Distance-to-Default Pipeline v{__version__}")
        logger.info(f"Python {sys.version.split()[0]} on {platform.platform()}")
        logger.info(f"Pandas {pd.__version__}, NumPy {np.__version__}, SciPy {scipy.__version__}")
        logger.info("=" * 80)
        
        # Check if we're just listing tickers
        if hasattr(args, 'list_tickers') and args.list_tickers:
            try:
                # Load the input file
                df = pd.read_excel(args.input, sheet_name=args.sheet)
                tickers = df['Instrument'].dropna().unique()
                print("\nTickers found in input file:")
                for i, ticker in enumerate(sorted(tickers), 1):
                    print(f"{i}. {ticker}")
                print(f"\nTotal: {len(tickers)} tickers")
                return 0
            except Exception as e:
                logger.error(f"Error listing tickers: {str(e)}")
                return 1
                
        # Check if we're just checking dependencies
        if hasattr(args, 'check_dependencies') and args.check_dependencies:
            try:
                check_requirements(logger)
                logger.info("All required dependencies are installed and up to date.")
                return 0
            except (RuntimeError, ImportError) as e:
                logger.error(str(e))
                return 1
                
        # Check if we're just showing the config
        if hasattr(args, 'show_config') and args.show_config:
            logger.info("Current configuration:")
            logger.info(f"  Input file: {args.input}")
            logger.info(f"  Output file: {args.output}")
            logger.info(f"  Sheet name: {args.sheet}")
            logger.info(f"  Log level: {args.log_level}")
            logger.info(f"  Test mode: {getattr(args, 'test', False)}")
            logger.info(f"  Max tickers: {getattr(args, 'max_tickers', 'No limit')}")
            return 0
            
        # Check if we're in test mode
        if hasattr(args, 'test') and args.test:
            if not hasattr(args, 'max_tickers') or not args.max_tickers:
                args.max_tickers = 3  # Default to 3 tickers in test mode
            if not hasattr(args, 'years'):
                args.years = 1  # Only 1 year of data in test mode
            logger.info(f"Running in test mode with max {args.max_tickers} tickers and {args.years} year of data")
        
    except Exception as e:
        logger.critical("=" * 80)
        logger.critical("CRITICAL ERROR: Pipeline execution failed")
        logger.critical("-" * 80)
        logger.critical(f"Error: {str(e)}", exc_info=True)
        logger.critical("=" * 80)
        return 1
    
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Log startup information
        logger.info("=" * 80)
        logger.info("Starting Distance-to-Default pipeline")
        logger.info(f"Command level: {args.log_level}")
        logger.info(f"Command line arguments: {vars(args)}")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Pandas version: {pd.__version__}")
        logger.debug(f"NumPy version: {np.__version__}")
        logger.info("-" * 80)
        
        # Check for required packages
        try:
            check_requirements(logger)
        except ImportError as e:
            logger.error(str(e))
            return 1
        
        # Load environment and configuration
        risk_free_rate = load_environment(logger)
        
        # Override risk-free rate if specified
        if args.risk_free_rate is not None:
            risk_free_rate = args.risk_free_rate
            logger.info(f"Using risk-free rate from command line: {risk_free_rate:.4f}")
        else:
            logger.info(f"Using risk-free rate from .env: {risk_free_rate:.4f}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        logger.info(f"Loading data from {args.input}")
        bank_data = load_and_prepare_data(
            args.input, 
            sheet_name=args.sheet,
            start_date=args.start_date, 
            end_date=args.end_date,
            logger=logger
        )
        
        # Get unique instruments
        instruments = bank_data['Instrument'].unique().tolist()
        
        # Get date range from data if not specified
        start_date = args.start_date or bank_data['Date'].min().strftime('%Y-%m-%d')
        end_date = args.end_date or bank_data['Date'].max().strftime('%Y-%m-%d')
        
        logger.info(f"Processing data from {start_date} to {end_date}")
        
        # Initialize price data as None
        prices_df = None

        # 1) Try local file if provided
        if args.price_file:
            try:
                prices_df = load_price_file(args.price_file).loc[start_date:end_date]
                if not prices_df.empty:
                    logger.info("✅ Local price file loaded successfully")
                    logger.debug(f"Local price data shape: {prices_df.shape}")
                else:
                    logger.warning("⚠️ Local price file is empty")
            except Exception as e:
                logger.warning(f"⚠️ Local file load failed: {e}")

        # 2) Try Refinitiv (library + REST fallback)
        if prices_df is None or prices_df.empty:
            try:
                prices_df = fetch_prices_refinitiv(
                    instruments, 
                    start_date, 
                    end_date, 
                    token=os.getenv('REFINITIV_API_TOKEN')
                )
                if not prices_df.empty:
                    logger.info("✅ Refinitiv price fetch succeeded")
                    logger.debug(f"Refinitiv data shape: {prices_df.shape}")
                else:
                    logger.warning("⚠️ Refinitiv returned empty price data")
            except Exception as e:
                logger.warning(f"⚠️ Refinitiv fetch failed: {e}")

        # 3) Try Yahoo Finance as last resort
        if prices_df is None or prices_df.empty:
            try:
                # Fall back to Yahoo Finance
                logger.info("Falling back to Yahoo Finance")
                prices_df = fetch_prices_yahoo(instruments, start_date, end_date, logger=logger)
                if not prices_df.empty:
                    logger.info("✅ Yahoo Finance fetch succeeded")
                    logger.debug(f"Yahoo Finance data shape: {prices_df.shape}")
                else:
                    logger.warning("⚠️ Yahoo Finance returned empty price data")
            except Exception as e:
                logger.error(f"❌ Yahoo Finance fetch failed: {e}")

        # 4) Final check if we have any price data
        if prices_df is None or prices_df.empty or prices_df.isna().all().all():
            raise RuntimeError("❌ No price data available from local file, Refinitiv, or Yahoo")
            
        logger.info(f"Successfully retrieved price data for {len(prices_df.columns)} instruments")
        
        # Calculate returns
        logger.info("Calculating returns...")
        returns_df = calculate_returns(prices_df, min_observations=args.min_returns)
        
        if returns_df.empty:
            raise ValueError("No valid returns could be calculated")
            
        logger.info(f"Calculated returns for {len(returns_df.columns)} instruments")
        
        # Process bank data and calculate KMV metrics
        logger.info("Calculating KMV metrics...")
        kmv_results = process_bank_data(
            bank_data, 
            returns_df, 
            risk_free_rate, 
            T=args.horizon,
            min_returns=args.min_returns
        )
        
        if kmv_results.empty:
            raise ValueError("No valid KMV results generated")
            
        logger.info(f"Successfully calculated KMV metrics for {len(kmv_results)} bank/date combinations")
        
        # Handle dry run
        if args.dry_run:
            logger.info("Dry run completed successfully (no files were written)")
            logger.info("The following files would be created:")
            logger.info(f"  - {os.path.abspath(args.output)}")
            logger.info(f"  - {os.path.abspath('dd_pipeline.log')}")
            return 0
            
        # Export results
        logger.info(f"Exporting results to {args.output}")
        try:
            success = export_results(
                prices_df, 
                returns_df, 
                kmv_results, 
                args.output,
                dry_run=args.dry_run
            )
            
            if success:
                if args.dry_run:
                    logger.info("Pipeline completed successfully (dry run)")
                else:
                    logger.info("Pipeline completed successfully")
                    logger.info(f"Results saved to: {os.path.abspath(args.output)}")
                    logger.info(f"Log file: {os.path.abspath('dd_pipeline.log')}")
                return 0
            else:
                logger.error("Failed to export results")
                return 1
                
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}", exc_info=True)
            logger.error("Pipeline completed with errors")
            return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    """Main entry point for the script.
    
    Handles uncaught exceptions and ensures proper exit codes.
    """
    try:
        # Run the main function and capture the exit code
        exit_code = main()
        
        # Ensure exit_code is an integer
        if not isinstance(exit_code, int):
            exit_code = 0 if exit_code is None else 1
            
        # Exit with the appropriate status code
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        # Log the full traceback for debugging
        logger.critical("=" * 80)
        logger.critical("CRITICAL: Unhandled exception in pipeline")
        logger.critical("-" * 80)
        logger.critical(f"Type: {type(e).__name__}")
        logger.critical(f"Error: {str(e)}")
        logger.critical("\n" + "-" * 80)
        logger.critical("Traceback (most recent call last):")
        import traceback
        logger.critical(traceback.format_exc())
        logger.critical("=" * 80)
        
        # Print a user-friendly error message
        print("\n" + "!" * 80, file=sys.stderr)
        print("ERROR: An unexpected error occurred. Check the log file for details.", file=sys.stderr)
        print(f"Error: {str(e)}", file=sys.stderr)
        print("!" * 80 + "\n", file=sys.stderr)
        
        sys.exit(1)
