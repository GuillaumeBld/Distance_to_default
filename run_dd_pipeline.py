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

def calculate_volatility(returns_df, min_observations=50, trading_days_param=TRADING_DAYS):
    """Calculate annualized volatility from daily returns.
    
    Args:
        returns_df: DataFrame of daily returns
        min_observations: Minimum number of observations required
        trading_days_param: Number of trading days in a year for annualization.
        
    Returns:
        Series of annualized volatilities with NaN for insufficient data
    """
    logger.info(f"Calculating annualized volatilities using {trading_days_param} trading days.")
    
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
    annual_vol = daily_vol * np.sqrt(trading_days_param)
    
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
    """Calculate KMV metrics using an iterative numerical optimization.
    
    This function implements an iterative procedure to solve for asset value (V_A)
    and asset volatility (sigma_A) based on the Merton model.

    Args:
        E: Market value of equity.
        D: Face value of debt.
        r: Risk-free rate (annualized).
        sigma_E: Equity volatility (annualized, from market data).
        T: Time horizon in years (default: 1 year).
        
    Returns:
        Tuple of (V_asset, sigma_A, DD, PD, converged_status)
    """
    # Ensure logger is available, even if it's just the root logger
    # This is good practice if the function might be called from contexts where logger isn't explicitly passed
    # However, in this script, logger is typically set up in main and available globally or passed.
    # For this specific refactoring, assuming 'logger' is accessible as a global or module-level logger.
    # If not, it should be passed as an argument: def calculate_kmv_metrics(E, D, r, sigma_E, T=1.0, logger=None):
    # and then use if logger: logger.warning(...)
    
    # Max iterations for the outer loop and tolerance for sigma_A convergence
    max_iterations = 100
    tolerance = 1e-5 # Tolerance for asset volatility convergence
    fsolve_xtol = 1e-8 # Tolerance for the inner fsolve
    fsolve_maxfev = 500 # Max function evaluations for fsolve

    # Initial guess for asset value
    V_iter = E + D
    if V_iter <= 0: # Handle edge case of non-positive initial asset value
        logger.warning(f"Initial asset value V_iter = E+D = {V_iter:.2f} is non-positive. E={E}, D={D}")
        return np.nan, np.nan, np.nan, np.nan, False

    # Initial guess for asset volatility (sigma_A_iter)
    # Use observed equity volatility (sigma_E) to make an initial estimate for asset volatility
    # sigma_A_iter = sigma_E * (E / (V_iter + 1e-10)) # More standard initial guess
    sigma_A_iter = sigma_E * (E / (E + D + 1e-10)) # As per plan
    sigma_A_iter = max(sigma_A_iter, 0.05)  # Enforce minimum 5% volatility
    sigma_A_iter = min(sigma_A_iter, 3.0) # Enforce a reasonable maximum for initial guess (e.g. 300%)

    if sigma_E <= 1e-6 : # Handle very small or zero sigma_E
        logger.warning(f"Initial sigma_E ({sigma_E:.4f}) is very small. KMV may not be reliable.")
        # sigma_A_iter might become very small or zero, max(..., 0.05) handles this.

    converged_outer_loop = False
    sqrt_T = np.sqrt(T)

    try:
        for i in range(max_iterations):
            sigma_A_old = sigma_A_iter
            
            # Step b, c, d: Calculate d1_iter, d2_iter, E_estimated_iter
            # These are based on the current iteration's V_iter and sigma_A_iter
            # Add small epsilon to denominators to prevent division by zero
            if sigma_A_iter * sqrt_T < 1e-10: # Avoid division by zero if sigma_A_iter or T is tiny
                logger.warning(f"sigma_A_iter * sqrt_T is too small ({sigma_A_iter * sqrt_T:.2e}) in iteration {i+1}. V_iter={V_iter:.2f}, sigma_A_iter={sigma_A_iter:.4f}")
                # This might indicate issues with inputs or convergence path
                # Depending on desired behavior, could break or try to adjust sigma_A_iter
                # For now, will likely lead to NaN in d1/d2 and fsolve failure or non-convergence
                # Let fsolve handle it, or return failure if it's problematic.
                # If d1 calculation fails, the try-except for fsolve will catch it.
                pass # Allow to proceed, fsolve might still handle it or fail gracefully

            d1_iter = (np.log(V_iter / (D + 1e-10)) + (r + 0.5 * sigma_A_iter**2) * T) / (sigma_A_iter * sqrt_T + 1e-10)
            d2_iter = d1_iter - sigma_A_iter * sqrt_T
            
            # E_estimated_iter is the theoretical equity value based on V_iter and sigma_A_iter
            E_estimated_iter = V_iter * norm.cdf(d1_iter) - D * np.exp(-r * T) * norm.cdf(d2_iter)
            
            if E_estimated_iter < 1e-10: # If estimated equity is near zero, sigma_E_target might blow up
                logger.debug(f"E_estimated_iter ({E_estimated_iter:.2e}) is near zero in iteration {i+1}. V_iter={V_iter:.2f}, sigma_A_iter={sigma_A_iter:.4f}")
                # This could lead to instability. May need to cap V_iter / (E_estimated_iter + 1e-10) or handle.
                # For now, let it proceed, but it's a potential source of issues.
                pass


            # Step e: Calculate sigma_E_target_for_fsolve
            # This is the equity volatility implied by the current V_iter and sigma_A_iter.
            # This target sigma_E will be used in the fsolve step.
            sigma_E_target_for_fsolve = (V_iter / (E_estimated_iter + 1e-10)) * norm.cdf(d1_iter) * sigma_A_iter
            sigma_E_target_for_fsolve = max(sigma_E_target_for_fsolve, 1e-4) # Ensure target sigma_E is not too small
            sigma_E_target_for_fsolve = min(sigma_E_target_for_fsolve, 5.0)  # And not excessively large

            # Step f: Use fsolve to find V_new and sigma_A_new
            # kmv_equation aims to match E (observed market equity) and sigma_E_target_for_fsolve
            solution, infodict, ier, msg = fsolve(
                kmv_equation,
                [V_iter, sigma_A_iter],  # Initial guess for this fsolve step
                args=(E, D, r, sigma_E_target_for_fsolve, T), # Note: E is market E, sigma_E is the target
                xtol=fsolve_xtol,
                maxfev=fsolve_maxfev,
                full_output=1
            )
            
            # Step g: Check fsolve convergence
            if ier != 1:
                logger.warning(f"Inner fsolve did not converge in iteration {i+1}: {msg}. V_iter={V_iter:.2f}, sigma_A_iter={sigma_A_iter:.4f}, target_sigma_E={sigma_E_target_for_fsolve:.4f}")
                converged_outer_loop = False
                break 
            
            # Step h: Update V_iter and sigma_A_iter with fsolve solution
            V_iter, sigma_A_iter = solution[0], solution[1]
            
            # Basic sanity checks for V_iter and sigma_A_iter from fsolve
            if V_iter <= 0 or sigma_A_iter <= 1e-3 or sigma_A_iter > 5.0: # Asset vol > 500% or too small
                logger.warning(f"Unreasonable solution from fsolve in iteration {i+1}: V={V_iter:.2f}, σA={sigma_A_iter:.4f}. Breaking.")
                converged_outer_loop = False
                break
            
            # Step i: Check for outer loop convergence (based on sigma_A_iter)
            if abs(sigma_A_iter - sigma_A_old) < tolerance:
                converged_outer_loop = True
                logger.debug(f"Converged in {i+1} iterations. V_A={V_iter:.2f}, sigma_A={sigma_A_iter:.4f}")
                break
            
            if i == max_iterations - 1:
                logger.warning(f"Outer loop reached max iterations ({max_iterations}) without converging. Last V={V_iter:.2f}, sigma_A={sigma_A_iter:.4f}, diff={abs(sigma_A_iter - sigma_A_old):.2e}")

        # After the loop
        if converged_outer_loop:
            # Calculate final DD and PD using converged V_iter and sigma_A_iter
            # Re-calculate d1 with final converged V_iter and sigma_A_iter
            if sigma_A_iter * sqrt_T < 1e-10: # Final check before d1 calculation
                 logger.warning(f"Final sigma_A_iter * sqrt_T is too small ({sigma_A_iter * sqrt_T:.2e}). V_iter={V_iter:.2f}, sigma_A_iter={sigma_A_iter:.4f}")
                 return np.nan, np.nan, np.nan, np.nan, False

            d1_final = (np.log(V_iter / (D + 1e-10)) + (r + 0.5 * sigma_A_iter**2) * T) / (sigma_A_iter * sqrt_T + 1e-10)
            # DD is d2, as per the problem's implied definition (PD = N(-DD) where DD = d2)
            DD = d1_final - sigma_A_iter * sqrt_T # This is d2
            PD = norm.cdf(-DD) # Corresponds to N(-d2)
            
            # Final check for reasonable values
            if not (np.isfinite(V_iter) and np.isfinite(sigma_A_iter) and np.isfinite(DD) and np.isfinite(PD)):
                logger.warning(f"Final KMV results are not finite: V={V_iter}, sigma_A={sigma_A_iter}, DD={DD}, PD={PD}")
                return np.nan, np.nan, np.nan, np.nan, False

            return V_iter, sigma_A_iter, DD, PD, True
        else:
            # If loop broke due to fsolve failure, non-convergence, or unreasonable values
            logger.warning(f"KMV calculation did not converge after iterations or failed. E={E:.2f}, D={D:.2f}, r={r:.4f}, input sigma_E={sigma_E:.4f}")
            return np.nan, np.nan, np.nan, np.nan, False
            
    except Exception as e:
        logger.error(f"Exception in calculate_kmv_metrics: {str(e)} for E={E}, D={D}, sigma_E={sigma_E}", exc_info=True)
        return np.nan, np.nan, np.nan, np.nan, False

def process_bank_data(bank_data, returns_df, risk_free_rate, T=1.0, min_returns=50, logger=None, trading_days_param=TRADING_DAYS):
    """Process bank data to calculate KMV metrics for each bank/date combination.
    
    Args:
        bank_data: DataFrame with bank information including dates
        returns_df: DataFrame with daily returns (index=date, columns=instruments)
        risk_free_rate: Annual risk-free rate (float)
        T: Time horizon in years (default: 1.0)
        min_returns: Minimum number of returns required for volatility calculation
        logger: Optional logger instance for logging messages
        trading_days_param: Number of trading days in a year for annualization.
        
    Returns:
        DataFrame with KMV metrics for each bank/date combination
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Processing KMV metrics with T={T} years and {trading_days_param} trading days for annualization.")
    
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

            # Calculate equity volatility (annualized) using a rolling window
            if ticker not in returns_df.columns:
                logger.warning(f"No returns data for {ticker} in returns_df")
                continue

            # Access the full returns series for the ticker
            ticker_returns = returns_df[ticker]

            # Find the index for the current date or the closest preceding date
            # Ensure returns_df.index is sorted (should be by design)
            try:
                # Get the position of the closest date <= current 'date'
                end_idx_loc = ticker_returns.index.get_indexer([date], method='ffill')[0]
                # Handle case where date is before the first date in returns
                if end_idx_loc == -1 : # date is before the start of the returns series
                    logger.warning(f"Date {date.date()} is before the start of returns data for {ticker}")
                    continue
                
                actual_end_date = ticker_returns.index[end_idx_loc]
                if actual_end_date > date: # Should not happen with ffill, but as a safeguard
                    logger.warning(f"Mismatch: Actual end date {actual_end_date.date()} for rolling window is after current date {date.date()} for {ticker}. Skipping.")
                    continue

            except KeyError: # Should not happen if returns_df.index is DatetimeIndex
                logger.warning(f"Current date {date.date()} not found and no preceding date for {ticker} in returns data.")
                continue
            except IndexError:
                logger.warning(f"IndexError while trying to find end date for rolling window for {ticker} on {date.date()}. Skipping.")
                continue

            # Define the start index for the rolling window
            # Window is from max(0, end_idx_loc - TRADING_DAYS + 1) to end_idx_loc + 1
            start_idx_loc = max(0, end_idx_loc - TRADING_DAYS + 1)
            
            # Select the returns for the rolling window
            rolling_returns = ticker_returns.iloc[start_idx_loc : end_idx_loc + 1]
            
            # Check for minimum number of observations in the window
            # Using min_returns directly as per instructions, could also use TRADING_DAYS // 2 or similar
            if rolling_returns.count() < min_returns:
                logger.warning(f"Insufficient returns data in rolling window for {ticker} on {date.date()} "
                              f"(have {rolling_returns.count()}, need {min_returns}) "
                              f"Window: {ticker_returns.index[start_idx_loc].date()} to {ticker_returns.index[end_idx_loc].date()}")
                continue
            
            # Calculate annualized standard deviation
            rolling_std = rolling_returns.std()
            sigma_E = rolling_std * np.sqrt(trading_days_param)

            if sigma_E <= 0 or not np.isfinite(sigma_E):
                logger.warning(f"Invalid rolling sigma_E={sigma_E:.6f} for {ticker} on {date.date()} (std: {rolling_std:.6f}), using {trading_days_param} days.")
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

def export_results(prices_df, returns_df, kmv_results, validation_results_df, output_file, dry_run=False):
    """Export results to Excel file with proper formatting.
    
    Args:
        prices_df: DataFrame with daily prices.
        returns_df: DataFrame with daily returns.
        kmv_results: DataFrame with KMV metrics.
        validation_results_df: Optional DataFrame with validation analysis results.
        output_file: Path to output Excel file.
        dry_run: If True, don't actually write any files.
        
    Returns:
        bool: True if export was successful or if dry_run is True.
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

            # Export Validation Analysis if provided
            if validation_results_df is not None and not validation_results_df.empty:
                validation_results_df.to_excel(writer, sheet_name='Validation_Analysis', index=False)
                worksheet_val = writer.sheets['Validation_Analysis']
                
                # Format header for Validation_Analysis
                for col_num, value in enumerate(validation_results_df.columns.values):
                    worksheet_val.write(0, col_num, value, header_format)
                
                # Apply formatting to Validation_Analysis columns
                for col_num, col_name in enumerate(validation_results_df.columns):
                    max_len = max(validation_results_df[col_name].astype(str).map(len).max(skipna=True), len(col_name)) + 2
                    worksheet_val.set_column(col_num, col_num, min(max_len, 30))

                    if 'Date' in col_name or 'Window_Start_Actual' in col_name or 'Window_End_Actual' in col_name :
                        worksheet_val.set_column(col_num, col_num, 12, date_fmt)
                    elif 'PD' in col_name:
                        worksheet_val.set_column(col_num, col_num, 15, pct_fmt)
                    elif 'DD' in col_name:
                         worksheet_val.set_column(col_num, col_num, 15, float_fmt)
                    elif 'Num_Observations' in col_name or 'Window_Months_Prior' in col_name:
                        worksheet_val.set_column(col_num, col_num, 18, int_fmt) # Integer format for counts
            else:
                logger.info("No validation analysis data to export or it was empty.")

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
    exec_group.add_argument(
        '--default-events-file',
        type=str,
        help='Optional CSV/Excel file with known default events (columns: Instrument, Default_Date)'
    )
    
    return parser.parse_args()

def validate_model_with_defaults(kmv_results_df, default_events_file_path, logger):
    """
    Validates the KMV model outputs against a list of known default events.

    Args:
        kmv_results_df (pd.DataFrame): DataFrame with KMV results (PD, DD).
                                       Must contain 'Instrument', 'Date', 'Probability_of_Default', 'Distance_to_Default'.
        default_events_file_path (str): Path to the CSV or Excel file containing default events.
                                        Must contain 'Instrument' and 'Default_Date'.
        logger (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame: A DataFrame summarizing PD/DD trends leading up to default events,
                      or None if validation cannot be performed.
    """
    if not default_events_file_path:
        logger.info("No default events file provided. Skipping validation.")
        return None

    logger.info(f"Loading default events from: {default_events_file_path}")
    try:
        if default_events_file_path.lower().endswith(('.xls', '.xlsx')):
            default_events_df = pd.read_excel(default_events_file_path)
        elif default_events_file_path.lower().endswith('.csv'):
            default_events_df = pd.read_csv(default_events_file_path)
        else:
            logger.error(f"Unsupported file format for default events: {default_events_file_path}. Please use CSV or Excel.")
            return None
    except FileNotFoundError:
        logger.error(f"Default events file not found: {default_events_file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading default events file {default_events_file_path}: {str(e)}")
        return None

    # Validate required columns
    required_cols = ['Instrument', 'Default_Date']
    if not all(col in default_events_df.columns for col in required_cols):
        logger.error(f"Default events file must contain columns: {', '.join(required_cols)}")
        return None

    # Prepare default events data
    try:
        default_events_df['Default_Date'] = pd.to_datetime(default_events_df['Default_Date'])
    except Exception as e:
        logger.error(f"Error parsing 'Default_Date' in default events file: {str(e)}")
        return None
    
    # Prepare KMV results data (ensure 'Date' is datetime)
    if 'Date' not in kmv_results_df.columns or not pd.api.types.is_datetime64_any_dtype(kmv_results_df['Date']):
        logger.error("'Date' column in KMV results is missing or not datetime. Cannot perform validation.")
        return None
    if not all(col in kmv_results_df.columns for col in ['Instrument', 'Probability_of_Default', 'Distance_to_Default']):
        logger.error("KMV results DataFrame is missing required columns for validation: 'Instrument', 'Probability_of_Default', 'Distance_to_Default'.")
        return None


    # Merge KMV results with default events
    # Ensure 'Instrument' types are compatible for merging, e.g., both strings.
    kmv_results_df['Instrument'] = kmv_results_df['Instrument'].astype(str)
    default_events_df['Instrument'] = default_events_df['Instrument'].astype(str)
    
    merged_df = pd.merge(kmv_results_df, default_events_df, on='Instrument', how='inner')

    if merged_df.empty:
        logger.warning("No common instruments found between KMV results and default events. Validation cannot proceed.")
        return pd.DataFrame() # Return empty DataFrame

    # Filter for records where KMV 'Date' is before 'Default_Date'
    relevant_data = merged_df[merged_df['Date'] < merged_df['Default_Date']].copy() # Use .copy() to avoid SettingWithCopyWarning

    if relevant_data.empty:
        logger.info("No KMV data found prior to default dates for the matched instruments.")
        return pd.DataFrame()

    # Define time windows before default (in months)
    windows_months = [12, 6, 3, 1] # e.g., 12 months prior, 6 months prior, etc.
    # More precise windows: e.g. (12,9), (9,6), (6,3), (3,1) months before default
    # For this implementation, let's consider data *within* X months of default.
    # Example: for 12 months window, data from (Default_Date - 12 months) to Default_Date
    
    validation_summary = []

    for instrument, group in relevant_data.groupby('Instrument'):
        default_date = group['Default_Date'].iloc[0] # Should be the same for all rows in the group
        
        for months_prior in windows_months:
            window_end_date = default_date
            window_start_date = default_date - pd.DateOffset(months=months_prior)
            
            # Filter data within this specific window
            window_data = group[
                (group['Date'] >= window_start_date) & (group['Date'] < window_end_date)
            ]
            
            if not window_data.empty:
                avg_pd = window_data['Probability_of_Default'].mean()
                median_pd = window_data['Probability_of_Default'].median()
                avg_dd = window_data['Distance_to_Default'].mean()
                median_dd = window_data['Distance_to_Default'].median()
                num_obs = len(window_data)

                validation_summary.append({
                    'Instrument': instrument,
                    'Default_Date': default_date,
                    'Window_Months_Prior': months_prior, # Signifies data up to X months before default
                    'Window_Start_Actual': window_data['Date'].min(),
                    'Window_End_Actual': window_data['Date'].max(),
                    'Avg_PD': avg_pd,
                    'Median_PD': median_pd,
                    'Avg_DD': avg_dd,
                    'Median_DD': median_dd,
                    'Num_Observations': num_obs
                })
            else:
                 validation_summary.append({
                    'Instrument': instrument,
                    'Default_Date': default_date,
                    'Window_Months_Prior': months_prior,
                    'Window_Start_Actual': pd.NaT,
                    'Window_End_Actual': pd.NaT,
                    'Avg_PD': np.nan,
                    'Median_PD': np.nan,
                    'Avg_DD': np.nan,
                    'Median_DD': np.nan,
                    'Num_Observations': 0
                })


    if not validation_summary:
        logger.info("No data found within any validation window for any defaulted firm.")
        return pd.DataFrame()
        
    summary_df = pd.DataFrame(validation_summary)
    
    # Sort for better readability
    summary_df = summary_df.sort_values(by=['Instrument', 'Default_Date', 'Window_Months_Prior'], ascending=[True, True, False])
    
    logger.info(f"Generated validation summary with {len(summary_df)} entries.")
    return summary_df


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
            T=1.0, # Defaulting to T=1.0 year, args.horizon was not defined
            min_returns=args.min_returns,
            logger=logger, # Pass the logger
            trading_days_param=args.trading_days # Pass the command-line arg
        )
        
        if kmv_results.empty:
            logger.warning("No valid KMV results generated. Skipping further processing.")
            # Depending on desired behavior, might exit or continue to try to export empty/partial results
        else:
            logger.info(f"Successfully calculated KMV metrics for {len(kmv_results)} bank/date combinations")

        # Perform model validation if default events file is provided
        validation_summary_df = None # Initialize as None
        if hasattr(args, 'default_events_file') and args.default_events_file:
            if not kmv_results.empty:
                logger.info("Performing model validation with default events...")
                validation_summary_df = validate_model_with_defaults(
                    kmv_results,
                    args.default_events_file,
                    logger
                )
                if validation_summary_df is not None and not validation_summary_df.empty:
                    logger.info(f"Generated validation summary with {len(validation_summary_df)} entries.")
                elif validation_summary_df is not None: # Empty dataframe
                     logger.info("Validation analysis did not yield any results (e.g. no matching firms or data in windows).")
                else: # None was returned
                    logger.warning("Validation analysis was skipped or failed.")
            else:
                logger.warning("KMV results are empty, skipping validation with default events.")
        else:
            logger.info("No default events file specified, skipping validation step.")

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
            # Ensure kmv_results is a DataFrame even if empty for export_results consistency
            if kmv_results is None: # Should ideally not happen if initialized properly
                kmv_results = pd.DataFrame()
                
            success = export_results(
                prices_df, 
                returns_df, 
                kmv_results, 
                validation_summary_df, # Pass the validation summary
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
