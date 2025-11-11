# Binder Design with anititargets

The `binder_antitarget` protocol is a powerful tool for designing protein binders that are optimized to bind to a specific target while simultaneously being optimized *not* to bind to a set of anti-targets. This is useful for designing binders that are specific to a particular target, or for preventing binders from self-associating.

## Installation

To use the `binder_antitarget` protocol, you will need to install ColabDesign from this repository.

First, install JAX with CUDA support:
```bash
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then, clone this repository and install it in editable mode:
```bash
git clone https://github.com/sokrypton/ColabDesign.git
cd ColabDesign
pip install -e .
```

## How it works

The `binder_antitarget` protocol works by combining the standard `binder` protocol with an additional loss term that penalizes binding to the anti-targets. The anti-targets can be specified as PDB files, or as the special keyword "self" to prevent the binder from self-associating.

## Example

Here is a simple example of how to use the `binder_antitarget` protocol:

```python
from colabdesign import mk_af_model

# create a model
model = mk_af_model(protocol="binder_antitarget")

# prepare the inputs
model.prep_inputs(pdb_filename="1a2y.pdb",
                  target_chain="A",
                  binder_len=30,
                  antitargets="self,1a2y.pdb:B")

# design the binder
model.design_3stage()

# save the results
model.save_pdb("binder.pdb")
```

In this example, we are designing a binder that binds to chain A of the PDB file `1a2y.pdb`, but not to chain B of the same PDB file, or to itself.
