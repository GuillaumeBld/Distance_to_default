# Bank Risk Analysis with ESG Scores

## Project Overview
This project specifically evaluates default risk for banks by combining:
- Real financial data
- ESG scores (Environmental, Social, Governance)
- A proven mathematical model

## How It Works

### What We Measure
1. **Distance-to-Default**:
   - Indicates how close a bank is to default
   - Higher score = more stable institution

2. **Probability of Default**:
   - Converts Distance-to-Default to risk percentage
   - From 0% (no risk) to 100% (certain default)

3. **ESG Scores**:
   - Environmental assessment
   - Social performance
   - Governance quality

### Bank-Specific Analysis
- **Score > 3**: Very stable bank (e.g., systemically important banks)
- **Score 1-3**: Bank to monitor (moderate risk)
- **Score ≤ 1**: Troubled bank (requires immediate attention)

ESG scores reveal:
- How sustainable practices impact bank stability
- Sector-specific ESG strengths/weaknesses (E, S or G)
- Resilience to climate and social risks

## Required Data

### Input File
An Excel file containing:
- Basic financial information:
  - Asset values
  - Debt amounts
  - Stock prices
- ESG scores:
  - Environmental performance
  - Social impact
  - Governance quality

## Output Results

The program generates an Excel file with:
1. Original financial data
2. Calculated risk scores:
   - Distance-to-Default
   - Probability of Default
3. ESG scores
4. Combined risk/ESG analysis

## How to Use Results

### Simple Interpretation
- Safe bank: Green score (>3)
- Moderate risk: Orange score (1-3)
- High risk: Red score (<1)
