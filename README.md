# Cell-memory

Python Code
===================

The code has been tested successfully with versions of Python (at least 2.7) on Windows and macOS.

-------------------------
Using the framework
-------------------------

The subfolders contains codes to generate the results obtained by numerical simulations in the main and supplementary figures.

The subfolders contains codes to generate the results obtained by numerical simulations in the main and supplementary figures.

Polarity model     : Kymographs in Figure 1D, Figure 1 - figure supplement 1C-E, Figure 4- supplement figure 1A,B. 

Physical model of single-cell chemotaxis : Cell shapes in Figure 1E, Figure 4B, Figure 4 - figure supplement 1C.


Each folder has a file with name ending '_main.py'. This file combines different objects (functions) defined in linked files in the same folder. 

In the folder  /Cell-memory/Polarity model, the 'EgfrPtpSde_main.py' loads the model equations and parameters from 'Models.py' file. 

Excecution of Polarity model:
```python

model = EgfrPtp()
initial_condition = around_steadystate()
lbo = Periodic()
stimulus = single_gradient()

rd = ReactionDiffusion1D(model, initial_condition, lbo,stimulus)
rd.simulate()
```



