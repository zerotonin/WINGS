# W.I.N.G.S. - Wolbachia Infection Numerical Growth Simulation.

## Project Overview
This project simulates the population dynamics of beetles affected by the Wolbachia bacterium. Wolbachia influences beetle reproduction and survival through mechanisms like Cytoplasmic Incompatibility (CI), Male Killing, Increased Exploration Rate, and Increased Egg Laying. Our model uses a Monte Carlo approach to explore various scenarios of Wolbachia's effects on beetle populations.

## Features
- Simulates beetle movement using Levy flights.
- Models Wolbachia's effects: CI, Male Killing, Increased Exploration, and Increased Egg Laying.
- Tracks population size and infection rates over time.
- Supports multiprocessing for efficient simulation of numerous scenarios.

## Installation
To set up the simulation environment, clone this repository and install the required dependencies.

```bash
git clone https://github.com/zerotonin/WINGS.git
cd WINGS
pip install -r requirements.txt
```
## Usage
Run the simulation with the following command:

```bash
python run_simulation.py

```

## Simulation Parameters
- **'GRID_SIZE'**: Size of the simulation grid.
- **'INITIAL_POPULATION'**: Initial number of beetles.
- **'INFECTED_FRACTION'**: Fraction of the initial population that is infected with Wolbachia.

## Contributing
We welcome contributions to this project! Please read our Contributing Guidelines for details on how to contribute.

## License
This project is licensed under the MIT License. **See LICENSE.md** for more details.

## Contact
For questions or feedback, please contact Bart Geurten.

### Notes

- **GitHub URL:** https://github.com/zerotonin/WINGS
- **Additional Details:**  This model is part of scientific publication **ADD LINK**. Please cite: **ADD CITATION HERE**


