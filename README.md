# UE5-FireImpact-Fracture-Framework
An Unreal Engine-Python framework for interactive thin-shell fracture simulation, integrating fire-induced material weakening and thermal damage history into real-time impact-driven destruction.

<img width="4178" height="700" alt="1" src="https://github.com/user-attachments/assets/a9014a17-de6e-46f7-bcc1-514175eb864b" />






## Overview
Conventional real-time fracture methods often fail to account for the influence of thermal environments, determining fragments based only on instantaneous impact forces. FireImpact-Fracture addresses this by modeling how structures exposed to flames progressively weaken over time. By representing thermal damage through a lightweight vertex-based weight model, the system generates asymmetric and multi-scale fracture patterns that reflect the prior fire exposure history in real time.

## Features
+ Fire-Induced Material Weakening
  + Tracks heat diffusion and combustion state transitions to modulate local materal strength.
 
+ Asymmetric Fracture Patterns
  + Cracks preferentially initiate and propagate along thermally weakened paths rather than just around the impact point.

+ Modular Dual-Architecture
  + Separate high-perfomance visualization(UnrealEngine/C++) from flexible numerial computation(Python).

+ Vertex-Centric Data Flow
  + Manages temperature, fuel, and material weights at the vertex level for stable simulation continuity.
 
+ Real-Time Perfomance
  + Achieves stable destruction effects without expensive thermo-mechanical coupling or finite element analysis.
 

## Pipeline


## Requirement
+ OS: Windows 10/11
+ UnrealEngine 5.0.3
+ Visual Studio 2022
+ Python 3.13.12
+ You need to set **conda environment** (https://github.com/sgsellan/fracture-modes)

## Installation
1. Set conda environment
2. Open anaconda prompt and activate fracture-modes ((https://github.com/sgsellan/fracture-modes))
3. Activate python code - **Unreal_tcp.py**.
   This is how to create a tcp communication network connecting Python and Unreal.
4. Open Unreal Project
5. Press play button.
   *Please make sure that the procedural mesh blueprint object and tcp_table object are at the level.
6. Run and Test
   You can now run a simulation that catches fire, spreads, and then destroys it.

## Results
