# Optimization of Cybersecurity Risk in Supply Chain Networks

This is a repo of my code for my Master's Thesis Project on "Optimization of Cybersecurity Risk in Supply Chain Networks".

The code is based on the following papers by T. Sawik:
1. [A linear model for optimal cybersecurity investment in Industry 4.0 supply chains](https://www.tandfonline.com/doi/pdf/10.1080/00207543.2020.1856442)

2. [Balancing cybersecurity in a supply chain under direct and indirect cyber risks](https://www.tandfonline.com/doi/pdf/10.1080/00207543.2021.1914356)

3. [A rough cut cybersecurity investment using
portfolio of security controls with maximum
cybersecurity value](https://www.tandfonline.com/doi/pdf/10.1080/00207543.2021.1994166)

I have extended the models by implemented a Directed Acyclic Graph (DAG) based network modelling approach on top of the pre-existing models presented by Sawik. The code is written in python.

## Details of each notebook

1. **MTP1.ipynb:** Cybsec_L, Cybsec_BW, Cyberport_SLP and Cyberport_UBP implemented using PuLP

2. **MTP2.ipynb:** SCybsec_L(Pmax), SCybsec_L(Lmax), SCybsec_L(Qmin) and SCybsec_L(Smin) implemented using PuLP

3. **MTP.ipynb:** All 8 models implemented using Gurobi 

4. **final_code.py:** Gurobi implementation from MTP.ipynb in a single python file

5. **DAG.ipynb:** A notebook trying different propagation constant algorithms on Cybsec_L

6. **MTP_final_code.ipynb:** Complete notebook with all code and plots, including comparisons

7. **paper_code_5_nodes.ipynb:** Notebook with the final code implemented on a 5 node graph for faster computation and results


## Disclaimer

Some code may be incomplete, work-in-progress or broken. Run at your own risk.
