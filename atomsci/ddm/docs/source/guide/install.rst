.. _install:

Installation
============

Prerequisites
-------------
**AMPL** is a Python 3 package that has been developed and run in a specific conda environment. The following prerequisites are necessary to install **AMPL**:

* conda (Anaconda 3 or Miniconda 3, Python 3)
 
Install
-------
Clone the git repository
^^^^^^^^^^^^^^^^^^^^^^^^
::

    git clone https://github.com/ATOMconsortium/AMPL.git
 
Create conda environment
^^^^^^^^^^^^^^^^^^^^^^^^
::

    cd conda
    conda create -y -n atomsci --file conda_package_list.txt
    conda activate atomsci
    pip install -r pip_requirements.txt

.. note::
   
    Depending on system performance, creating the environment can take some time.

Install AMPL
------------
Go to the AMPL root directory and install the AMPL package::

    conda activate atomsci
    cd ..
    ./build.sh && ./install.sh system

* The `install.sh` system command installs AMPL directly in the conda environment. If `install.sh` alone is used, then AMPL is installed in the `$HOME/.local` directory.

* After this process, you will have an `atomsci` conda environment with all dependencies installed. The name of the AMPL package is `atomsci-ampl` and is installed in the `install.sh` script to the environment with conda's pip.  

Install with Docker
-------------------
* Download and install Docker Desktop.
   * `Docker Getting Started <https://www.docker.com/get-started>`_
* Create a workspace folder to mount with Docker environment and transfer files.
* Get the Docker image and run it::

    docker pull paulsonak/atomsci-ampl
    docker run -it -p 8888:8888 -v </local_workspace_folder>:</directory_in_docker> paulsonak/atomsci-ampl
    #inside docker environment
    jupyter-notebook --ip=0.0.0.0 --allow-root --port=8888 &
    # -OR-
    jupyter-lab --ip=0.0.0.0 --allow-root --port=8888 &

* Visit the provided URL in your browser, ie
   * http://d33b0faf6bc9:8888/?token=656b8597498b18db2213b1ec9a00e9d738dfe112bbe7566d
   * Replace the d33b0faf6bc9 with localhost
   * If this doesn't work, exit the container and change port from 8888 to some other number such as 7777 or 8899 (in all 3 places it's written), then rerun both commands

* Be sure to save any work you want to be permanent in your workspace folder. If the container is shut down, you'll lose anything not in that folder.

