# SAA-and-TAA-potfolio-simulation

## Project Overview

This project was part of Quantitative Asset and Risk Management course at the University of Lausanne (UNIL), Master of Science in Finance program.

## Table of Contents

- [Project Overview](#project-overview)
- [Investment Universe](#investment-universe)
- [Project Steps](#project-steps)
  - [Part 1: Strategic Asset Allocation (SAA)](#part-1-strategic-asset-allocation-saa)
  - [Part 2: Tactical Asset Allocation (TAA)](#part-2-tactical-asset-allocation-taa)
  - [Part 3: Implementation](#part-3-implementation)
- [Data](#data)
- [Results and Analysis](#results-and-analysis)
  - [SAA Performance](#saa-performance)
  - [TAA Performance](#taa-performance)
  - [Implementation Performance](#implementation-performance)
- [Conclusion](#conclusion)

## Investment Universe

The investment universe includes the following asset classes:

- World equities
- World bonds
- US investment grade bonds
- US high yield bonds
- Gold
- Energy commodities
- Industrial Metals

## Project Steps

### Part 1: Strategic Asset Allocation (SAA)

1. **Create SAA (December 2010)**
   - Use data available up to December 2010.
   - Design a systematic investment process.
   - Select at least three asset classes including world bonds and US investment grade bonds.
   - Explain the allocation scheme and asset class selection.
   - Show in-sample performance with charts and performance figures (annualized performance, volatility, Sharpe ratio, maximum drawdown, hit ratio).

2. **Analyze Out-of-Sample Performance**
   - Fast forward to the last date of the sample.
   - Compute performance metrics and compare them with the benchmark (50% equities and 50% bonds, rebalanced monthly).
   - Highlight strengths and weaknesses with charts comparing SAA and the benchmark.

### Part 2: Tactical Asset Allocation (TAA)

1. **Create TAA (December 2010)**
   - Use data available up to December 2010.
   - Create a long-short value factor based on z-scores from the "carry" table.
   - Allocate 2% ex ante volatility budget equally to longs and shorts, scaling the portfolio accordingly.
   - Chart historical cumulated performance and comment.
   - Create a long-short momentum factor using past 12-month performance with the same volatility budget.
   - Explain the rationale for using long-short factors.

2. **Analyze Performance Based on VIX Level**
   - Analyze the performance of value and momentum factors based on standardized VIX levels.
   - Devise a dynamic tilt based on findings (parametric allocation, FLAM allocation, or own invention).
   - Compute the expected Information Ratio (IR) for the tilt.

3. **Out-of-Sample Performance Analysis**
   - Compute performance metrics for the investment solution.
   - Compare with the benchmark, highlighting strengths and weaknesses.
   - Provide charts and tables summarizing the added value of the overlay.

### Part 3: Implementation

1. **Real-Life Implementation**
   - Adjust the portfolio model to exclude US investment grade bonds.
   - Explain the replication risk and its management.
   - Run a replication of the full portfolio model (SAA + TAA without US investment grade) from December 2010 onwards.
   - Compute ex ante and ex post tracking errors.
   - Perform performance attribution for replication errors based on sectors (bonds, equities, commodities).
   - Analyze the deviation from the model portfolio (allocation vs. selection).

## Data

The dataset contains monthly returns of the specified asset classes from 2000 to 2021. The data is available in the accompanying Excel file.

## Results and Analysis

### SAA Performance

#### In-Sample Performance

The ERC strategy provided the following weights for each asset:

- World Bonds: 58.44%
- US Investment Grade Bonds: 12.66%
- US High Yield Bonds: 8.53%
- Gold: 6.55%
- World Equities: 6.09%
- Industrial Metals: 4.15%
- Energy Commodities: 3.58%

Performance metrics compared to the benchmark:

| Metric                | SAA        | Benchmark  |
|-----------------------|------------|------------|
| Mean                  | 0.0645     | 0.0328     |
| Volatility            | 0.0427     | 0.0817     |
| Maximum Drawdown      | -0.111     | -0.29      |
| Sharpe Ratio          | 1.548      | 0.4011     |
| Hit Ratio             | 0.7559     | 0.59016    |

#### Out-of-Sample Performance

| Metric                | SAA        | Benchmark  |
|-----------------------|------------|------------|
| Mean                  | 0.0366     | 0.0632     |
| Volatility            | 0.0373     | 0.0672     |
| Maximum Drawdown      | -0.049     | -0.1015    |
| Sharpe Ratio          | 0.980      | 0.940      |
| Hit Ratio             | 0.62.097   | 0.68548    |

Despite underperforming the benchmark in terms of returns out-of-sample, the SAA achieved a lower volatility, resulting in a higher Sharpe ratio.

### TAA Performance

#### Value and Momentum Strategies

- The value strategy underperformed compared to the SAA in-sample, with higher variations in returns.
- The momentum strategy added a strong positive performance tilt to the SAA strategy, especially during the 2008 crisis.

#### VIX-Based Analysis

- Returns tended to be higher when the market was either very stable or very dynamic.
- The target portfolio showed an Information Ratio (IR) of 0.0285.

#### Out-of-Sample Performance

| Metric                | Target     | Benchmark  |
|-----------------------|------------|------------|
| Mean                  | 0.0546     | 0.0632     |
| Volatility            | 0.0954     | 0.0672     |
| Maximum Drawdown      | -0.0247    | -0.1015    |
| Sharpe Ratio          | 0.5723     | 0.939      |
| Hit Ratio             | 0.7479     | 0.68548    |

### Implementation Performance

- Excluding US investment grade bonds resulted in a tracking error mean of 6.8%.
- Performance attribution showed that the deviation from the model was primarily due to selection effects in the Bonds sector.
- The replication risk was managed by minimizing the variance of the tracking error for each date.
- Adjustments were made primarily in world equities and gold to compensate for the exclusion of US investment grade bonds.
- Allocation effects and selection effects were analyzed, showing that the deviation from the model portfolio was mainly due to selection effects.

| Metric                | Target Portfolio | Benchmark  |
|-----------------------|------------------|------------|
| Mean                  | 0.0546           | 0.0632     |
| Volatility            | 0.0954           | 0.0672     |
| Maximum Drawdown      | -0.0247          | -0.1015    |
| Sharpe Ratio          | 0.5723           | 0.939      |
| Hit Ratio             | 0.7479           | 0.68548    |

The following chart shows the cumulative returns of the target portfolio compared to the benchmark:

## Conclusion

Although the ERC portfolio did not outperform the benchmark initially, combining SAA and TAA with a methodical strategy based on the VIX index allowed us to come very close. The exercise highlighted the importance of adapting strategies based on market conditions and third-party information. It was also important to manage replication risk effectively when certain asset classes were excluded from the investment universe. Overall, this project provided valuable insights into developing and implementing a robust multi-asset investment process.

---

This README.md provides a comprehensive overview of the project, including detailed results and analysis from the SAA, TAA, and implementation phases.
