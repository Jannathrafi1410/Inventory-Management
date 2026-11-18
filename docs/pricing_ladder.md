# Pricing Optimization and Markdown Ladder

- **Note:** Simulator outputs under stated assumptions, not TBO real P&L. Demand shifted by (p/p0)^epsilon with segment elasticities; dynamic cost pass-through tau=0.30 on c_d; q*(p) unconstrained fractile-capped re-solve per grid point.
- **Price grid:** 0.88x to 1.16x of p0 (15 points) | **Scenarios:** 120 | **Seed:** 42

## Optimal price per cell (argmax expected profit) and flatness

| cell | segment | epsilon | p0 (INR) | argmax price (INR) | argmax q (units) | argmax E[profit] (INR) | max loss vs opt (%) | grid within 1% of opt (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DXB-NYE | event | -0.80 | 21000 | 24360 | 113.84 | 997277.82 | 56.65% | 6.7% |
| DXB-DEC | leisure | -2.50 | 14500 | 16820 | 120.17 | 652606.93 | 48.30% | 20.0% |
| AIR-BOMDXB | mixed | -0.80 | 34000 | 39440 | 50.54 | 427751.20 | 119.18% | 6.7% |
| SIN-F1 | event | -0.80 | 26000 | 30160 | 69.09 | 730979.29 | 63.50% | 6.7% |
| MLE-DEC | leisure | -2.50 | 42000 | 48720 | 28.83 | 483231.31 | 72.09% | 6.7% |

## Markdown ladder (release-deadline +14/+7/+3 days)

Lowers price when marginal unit's expected margin falls below salvage (refund value per unit).

### DXB-NYE (q_committed=113.8 units, salvage=0 INR/unit)

| days to release | price (INR) | multiplier | P(sell marginal) | margin vs salvage (INR) | markdown needed |
| ---: | ---: | ---: | ---: | ---: | :---: |
| +14 | 18480 | 0.88 | 0.867 | 4320.33 | False |
| +7 | 18480 | 0.88 | 0.867 | 4320.33 | False |
| +3 | 18480 | 0.88 | 0.867 | 4320.33 | False |

### DXB-DEC (q_committed=120.2 units, salvage=6714 INR/unit)

| days to release | price (INR) | multiplier | P(sell marginal) | margin vs salvage (INR) | markdown needed |
| ---: | ---: | ---: | ---: | ---: | :---: |
| +14 | 12760 | 0.88 | 0.892 | -3889.60 | True |
| +7 | 12760 | 0.88 | 0.892 | -3889.60 | True |
| +3 | 12760 | 0.88 | 0.892 | -3889.60 | True |

### AIR-BOMDXB (q_committed=50.5 units, salvage=8850 INR/unit)

| days to release | price (INR) | multiplier | P(sell marginal) | margin vs salvage (INR) | markdown needed |
| ---: | ---: | ---: | ---: | ---: | :---: |
| +14 | 29920 | 0.88 | 0.925 | -8461.50 | True |
| +7 | 29920 | 0.88 | 0.933 | -8458.00 | True |
| +3 | 29920 | 0.88 | 0.942 | -8454.50 | True |

### SIN-F1 (q_committed=69.1 units, salvage=0 INR/unit)

| days to release | price (INR) | multiplier | P(sell marginal) | margin vs salvage (INR) | markdown needed |
| ---: | ---: | ---: | ---: | ---: | :---: |
| +14 | 22880 | 0.88 | 0.983 | 4798.67 | False |
| +7 | 22880 | 0.88 | 0.992 | 4839.33 | False |
| +3 | 22880 | 0.88 | 1.000 | 4880.00 | False |

### MLE-DEC (q_committed=28.8 units, salvage=22050 INR/unit)

| days to release | price (INR) | multiplier | P(sell marginal) | margin vs salvage (INR) | markdown needed |
| ---: | ---: | ---: | ---: | ---: | :---: |
| +14 | 36960 | 0.88 | 1.000 | -16590.00 | True |
| +7 | 36960 | 0.88 | 1.000 | -16590.00 | True |
| +3 | 36960 | 0.88 | 1.000 | -16590.00 | True |

