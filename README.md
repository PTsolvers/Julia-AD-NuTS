# Julia-AD-NuTS

[![NuTS](docs/nuts.png)](https://nuts-w2.sciencesconf.org) [**NuTS - workshop**](https://nuts-w2.sciencesconf.org)

**High-performance computing in geosciences and adjoint-based optimisation with Julia on GPUs**

| The computational resources for this workshop are provided by the      |                |
| :------------: | :------------: |
| [**GLiCID mesocentre**](https://www.glicid.fr) | [![NuTS](docs/glicid.png)](https://www.glicid.fr) |


## Program

### Part 1 (Thursday May 30, 14h - 18h)
- [Login to the GLiCID Jupyter Hub (and troubleshooting)](#login-to-the-glicid-jupyter-hub)
- [Brief **intro to Julia for HPC** :book:](julia-ad-nuts.ipynb)
  - Performance, CPUs, GPUs, array and kernel programming
- [Presentation of **the challenge of the workshop** :book:](julia-ad-nuts.ipynb)
  - Optimising injection/extraction from a heterogeneous reservoir
- [**Hands-on I** - solving the forward problem :computer:](julia-ad-nuts.ipynb)
  - Steady-state diffusion problem
  - The accelerated pseudo-transient method
  - From CPU to GPU array programming
  - Kernel programming (performance)
    - CPU "kernel" programming -> multi-threading
    - GPU kernel programming

### Part 2 (Friday May 31, 9h - 12h and 13h30 - 15h30)
- [Presentation of **the optimisation problem** :book:](julia-ad-nuts.ipynb)
  - Tha adjoint method
  - Julia and the automatic differentiation (AD) tools
- [**Hands-on II** - HPC GPU-based inversions :computer:](julia-ad-nuts.ipynb)
  - The adjoint problem and AD
  - GPU-based adjoint solver using [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
  - Sensitivity analysis
  - Gradient-based inversion (Gradient descent - GD)
- **Optional exercises** :computer:
  - Use [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) as optimiser
  - Going for 3D
  - Make combined loss (pressure + flux)
- **Wrapping up** & outlook :beer:

## The `SMALL` print
The goal of this workshop is to develop a fast iterative GPU-based solver for elliptic equations and use it to:
1. Solve a steady state subsurface flow problem (geothermal operations, injection and extraction of fluids)
2. Invert for the subsurface permeability having a sparse array of fluid pressure observations

We will not use any "black-box" tooling but rather try to develop concise and performant codes (300 lines of code, max) that execute on graphics processing units (GPUs). We will also use automatic differentiation (AD) capabilities and the differentiable Julia language to automatise the calculation of the adjoint solutions in the gradient-based inversion procedure.

The main Julia packages we will rely on are:
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) for GPU computing on Nvidia GPUs
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for AD on GPUs
- [CairoMakie.jl](https://github.com/MakieOrg/Makie.jl) for plotting
- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to extend the "vanilla" gradient-descent procedure

Most of the workshop is based on "hands-on". Changes to the scripts -Jupyter notebooks for this workshop- are incremental and should allow to build up complexity throughout the 2 days. Blanked-out notebooks for most of the steps are available in the [notebooks](notebooks/) folder. Solutions notebooks (following the `s_xxx.ipynb` pattern) will be shared at some point in the [notebooks_solutions](notebooks_solutions) folder. _(Script versions of the notebooks are available in the corresponding [scripts](scripts/) and the [scripts_solutions](scripts_solutions) folders.)_

#### :bulb: Useful extra resources
- The Julia language: [https://julialang.org](https://julialang.org)
- PDE on GPUs ETH Zurich course: [https://pde-on-gpu.vaw.ethz.ch](https://pde-on-gpu.vaw.ethz.ch)
- Julia Discourse (Julia Q&A): [https://discourse.julialang.org](https://discourse.julialang.org)
- Julia Slack (Julia dev chat): [https://julialang.org/slack/](https://julialang.org/slack/)

## Login to the GLiCID Jupyter Hub

To start, let's make sure that everyone can connect to the [GLiCID Jupyter Hub](https://nuts-workshop.glicid.fr/) in order to access GPU resources: [https://nuts-workshop.glicid.fr/](https://nuts-workshop.glicid.fr/)

To start the GLiCID Jupyter Hub, you should use the credentials you received in the second email (subject `Your account has been validated`) after having followed the account creation procedure (see attached PDF in the info mail from Friday 17.05.24)

If all went smooth, you should be able to see and execute the [notebooks/visu_2D.ipynb](notebooks/visu_2D.ipynb) notebook which will produce this figure:

![out visu](docs/out_visu_2D.png)

> Note: The Jupyter notebooks are generated automatically using [Literate.jl](https://github.com/fredrikekre/Literate.jl)-powered literate programming in Julia upon running the [deploy_notebooks.jl](deploy_notebooks.jl) script.
