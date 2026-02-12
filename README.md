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
The system operates through a synchronized three-layer architecture connected via TCP/IP:
### Step 1 - Real-Time Flame Simualtion (UE5 C++)
+ Heat Diffusion : Propagates heat along mesh connectivity
+ Combustion States : Vertices transition through states, Unburned->Heating->Igniting->Burning->BurnedOut.
+ Material Weight Update : Local strength decreases progressively during combusion.

### Step 2 - Commmunication & Trigger (TCP/IP)
+ When a collision is detected, the engine extracts vertex positions and material weights to transmit tem to the Python module.

### Step 3 - Fracture Computation (Python)
+ Impact Amplification : Modulates sensitivity using the material weight
+ The module solves for fragmentation and returns updated mesh topology (indices and attributes)to the engine.
  

## Requirement
+ OS: Windows 10/11
+ UnrealEngine 5.0.3
+ Visual Studio 2022
+ Python 3.13.12
+ You need to set **conda environment** (https://github.com/sgsellan/fracture-modes)
+ Unreal Plugin - 'TcpSocket'(https://www.fab.com/listings/48db4522-8a05-4b91-bcf8-4217a698339b)
  - 'Blueprint Array Helper' (https://www.fab.com/listings/87af7b30-1635-4321-b1df-6e1f459cf855)

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

<img width="4400" height="1035" alt="2" src="https://github.com/user-attachments/assets/9b6c0980-767c-4f23-b854-1930be44f5f5" />

<img width="2906" height="1307" alt="3" src="https://github.com/user-attachments/assets/b5ac9738-2466-4d3d-a2a0-65beb6c511a8" />

<img width="2754" height="1022" alt="4" src="https://github.com/user-attachments/assets/9c9e549b-c04d-4625-94f9-b6d8c7d883ce" />

+ Baseline vs. Ours
  - Unlike impact-only models that produce symmetric fragmentation , our method results in spatially biased patterns where regions close to the fire are finely shattered while distant regions remain in larger fragments.
