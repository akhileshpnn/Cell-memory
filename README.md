# Cell-memory

Python Code
===================

The code has been tested successfully with versions of Python (at least 2.7) on Windows and macOS.

-------------------------
Using the framework
-------------------------

The subfolders contain codes to generate the results obtained by numerical simulations in the eLife article,

[Cells use molecular working memory to navigate in changing chemoattractant fields](https://elifesciences.org/articles/76825)


Polarity model     : Kymographs in Figure 1D, Figure 1 - figure supplement 1D-H, Figure 4- figure supplement 1A,C, Figure 4- figure supplement 2A,C 

Physical model of single-cell chemotaxis : Cell shapes in Figure 1E, Figure 4B, Figure 4 - figure supplement 1B,D, Figure 4- figure supplement 2B,D


Each folder has a file with name ending '_main.py'. This file combines different objects (functions) defined in linked files in the same folder. 

For example: In the folder  /Cell-memory/Polarity model, the 'EgfrPtpSde_main.py' loads the model equations and parameters from 'Models.py' file. 

Excecution of Polarity model:
```python

model = EgfrPtp()
initial_condition = around_steadystate()
lbo = Periodic()
stimulus = single_gradient()

rd = ReactionDiffusion1D(model, initial_condition, lbo,stimulus)
rd.simulate()
```

Execution of Physical model of single-cell chemotaxis: 'LevelSet_main.py' file loads kymograph corresponding to Figure 1D and generate cell shapes in Figure 1E. A single CPU of a normal desktop computer (eg. with Intel64 Family 6 processor) requires ~24 hrs for completion.
