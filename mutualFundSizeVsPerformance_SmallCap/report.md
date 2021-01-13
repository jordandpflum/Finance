# Analysis of Mutual Fund Size on Performance for Small Cap Value Funds



Contributing Members:

- Jordan Pflum
- Shan Ali
- David Cruz
- Alishah Vidhani

### Introduction

It is no secret the largest mutual funds have a competitive advantage compared to the common retail investors. Their wealth of experience, vast stores/access to information, and virtually unlimited resources make them formidable market players. Although this logic makes intuitive sense, the question remains whether one can prove this empirically. 

This report analyzes the effect of mutual fund size' on their performance. We measure a mutual fund's size by their Total Net Assets and their performance by their return (measure in monthly time units). Additionally, we restrict the universe of mutual funds to a single strategy (Small Cap Value) as defined by the Lipper Classification to control for bias' that would be introduced with funds employing a different (perhaps less volatile) strategy. 

### Methodology

1. Pulled monthly TNA and Return Data from mutual funds from CRSP.
2. Pulled mutual fund classification based on Lipper Classification from CRSP - filtered fund strategies to only include 'SCVE' (Small Cap Value) according to instructions.
3. Monthly Analysis
   1. Reported Mutual Funds return's relationship vs their respective Total Net Assets.
4. Aggregate Monthly (Annualized) Analysis
   1. Reported the average Monthly Return (annualized) for mutual funds vs their respective Total Net Assets

### Results

![Alt text](Images/returnVsTNA_Monthly.png?raw=true "Return Vs TNA (monthly)")
![Alt text](Images/riskVsTNA_annualized.png?raw=true "Risk Vs TNA (monthly)")
![Alt text](Images/returnVsTNA_annualized.png?raw=true "Return Vs TNA (annualized)")
![Alt text](Images/riskVsTNA_annualized.png?raw=true "Risk Vs TNA (annualized)")

|           Model            | Coefficient  | Standard Error | P-Value |  R^2  |
| :------------------------: | :----------: | :------------: | :-----: | :---: |
|  TNA vs Return (monthly)   | 3.73605e-07  |   2.310e-07    |  0.106  |  0.0  |
| TNA vs Return (Annualized) | -1.6667e-06  |   1.030e-06    |  .107   | 0.002 |
|   TNA vs Risk (monthly)    | -3.37054e-06 |    2.17e-05    |  .102   |   0   |
|  TNA vs Risk (Annualized)  | -5.7736e-06  |    3.58e-06    |  0.107  | 0.002 |

### Conclusion

Surprisingly, we found the the Total Net Assets of a firm employing a Small Cap Value Strategy had no statistically significant impact on Expected Return (or Risk). However, it should be noted that there were limited number of 'Mega' Funds (funds with over $2B is Total Net Assets). Those funds appeared to have lower risk than smaller funds.

Given the findings of this report, the question can be posed if a fund's size, previously thought to be a tremendous strength, can possibly be their Achilles heel. Perhaps their size, and by extension their market impact, make profitable strategies unviable. Or maybe their large market orders are too easily identifiable by increasing sophisticated trading algorithms causing other players to undercut previously safe profits. 

Our group wished to analyze the effect of a mutual fund's size on their performance
