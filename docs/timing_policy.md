# Timing Policy (Tranching, Priced WAIT, Release)

- **Note:** Simulator outputs under stated assumptions, not TBO real P&L. Tranches 40/35/25% at -120/-75/-45d resized by current fractile; WAIT compares buy-now vs after-signal (sigma -10%, rate drift +0.5%/30d); release holds if P(demand arrives)*margin > refund value.

## DXB-NYE

### (a) Tranching policy
- Base fractile: 0.3246 | Base target: 120.00 units | Total: 120.00 units (1619400 INR)

| days before | planned share | current fractile | resized units | cumulative units | cash (INR) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -120 | 40% | 0.3246 | 48.00 | 48.00 | 647760 |
| -75 | 35% | 0.3246 | 42.00 | 90.00 | 566790 |
| -45 | 25% | 0.3246 | 30.00 | 120.00 | 404850 |

### (b) Priced WAIT action
- Recommendation: **WAIT** | value_wait = 24381.47 INR
- Rationale: E[profit|after next signal]=718974.93 INR vs E[profit|buy now]=694593.46 INR; expected rate drift x1.0050 over 30d; demand sigma reduction 10%.

### (c) Release-date logic
- Recommendation: **HOLD** | P(demand arrives)=0.967, margin=7505 INR/unit, refund=0 INR/unit, hold value=7254.83 INR/unit

## DXB-DEC

### (a) Tranching policy
- Base fractile: 0.5454 | Base target: 150.00 units | Total: 150.00 units (1438800 INR)

| days before | planned share | current fractile | resized units | cumulative units | cash (INR) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -120 | 40% | 0.5454 | 60.00 | 60.00 | 575520 |
| -75 | 35% | 0.5454 | 52.50 | 112.50 | 503580 |
| -45 | 25% | 0.5454 | 37.50 | 150.00 | 359700 |

### (b) Priced WAIT action
- Recommendation: **WAIT** | value_wait = 16191.72 INR
- Rationale: E[profit|after next signal]=615075.08 INR vs E[profit|buy now]=598883.36 INR; expected rate drift x1.0050 over 30d; demand sigma reduction 10%.

### (c) Release-date logic
- Recommendation: **RELEASE** | P(demand arrives)=0.958, margin=4908 INR/unit, refund=6714 INR/unit, hold value=4703.50 INR/unit

## AIR-BOMDXB

### (a) Tranching policy
- Base fractile: 0.1100 | Base target: 56.34 units | Total: 56.34 units (1662127 INR)

| days before | planned share | current fractile | resized units | cumulative units | cash (INR) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -120 | 40% | 0.1100 | 22.54 | 22.54 | 664851 |
| -75 | 35% | 0.1100 | 19.72 | 42.26 | 581744 |
| -45 | 25% | 0.1100 | 14.09 | 56.34 | 415532 |

### (b) Priced WAIT action
- Recommendation: **WAIT** | value_wait = 14242.91 INR
- Rationale: E[profit|after next signal]=174778.70 INR vs E[profit|buy now]=160535.78 INR; expected rate drift x1.0050 over 30d; demand sigma reduction 10%.

### (c) Release-date logic
- Recommendation: **RELEASE** | P(demand arrives)=1.000, margin=4500 INR/unit, refund=8850 INR/unit, hold value=4500.00 INR/unit

