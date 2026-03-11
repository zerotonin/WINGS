Installation
============

From PyPI (when published)
--------------------------

.. code-block:: bash

   pip install wings-sim

   # With GPU support
   pip install wings-sim[gpu]

From source (development)
-------------------------

.. code-block:: bash

   git clone https://github.com/zerotonin/WINGS.git
   cd WINGS
   pip install -e ".[dev,docs]"

   # With GPU support
   pip install -e ".[gpu,dev,docs]"

Conda environments
------------------

Pre-configured environments are provided in ``envs/``:

.. code-block:: bash

   # CPU only
   conda env create -f envs/wings_cpu.yml
   conda activate wings

   # GPU (CUDA 12.1, for HPC clusters)
   conda env create -f envs/wings_gpu.yml
   conda activate wings-gpu

Verify GPU installation:

.. code-block:: bash

   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
