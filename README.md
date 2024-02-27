Experimental research code for our paper "Unleashing Graph Partitioning for Large-Scale Nearest Neighbor Search".

The ```exp_scripts``` folder contains scripts to download the datasets (beware they are large and downloading MS Turing takes forever) and build the source code -- in a way that is expected by the experiment scripts, e.g., ```experiments.py```.

To run the experiments, point the ```data_path``` variable to the folder containing the datasets, and create a folder ```exp_outputs``` in the top-level.
Then run ```python3 experiments.py```, which will place results in csv format in the ```exp_outputs``` folder. A query and routing simulation with s = 40-60 shards on 1B points takes roughly 12 hours per partition. The largest fraction of this time is spent on building HNSW indices in the shards and building routing indices.

**Disclaimer**: This is hacky research code intended as a testbed -- not of industrial-grade quality. As such, there may be bugs and there are no warnings if something is misconfigured. If some results with this code look unexpected please contact me.