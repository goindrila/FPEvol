# FPEvol
FPEvol is a python script for Fokker-Planck (FP) evolution of ultrarelativistic pair beams undergoing energy loss and diffusion in momentum space
Currently the terms that are implemented are based on plasma instabilities, i.e, energy loss due to instability and beam broadening due to instability. Instabilities are assumed electrostatic at the moment.
The FP equations are solved using a finite difference upwind scheme for an 2D Gaussian input distribution and Lorentz factor of 100. The input parameters can be changed and relevant energy loss and diffusion terms can be added.
