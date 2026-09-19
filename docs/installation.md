# Installing it

```bash
pip install funcyflows              # torch + tqdm, that's it
```

`torch` is enormous and fussy about your hardware. If you need a specific CUDA or MPS build,
install that first from [pytorch.org](https://pytorch.org) and put FuncyFlows on top —
otherwise pip will pick something for you and you may not like it.


## From a checkout

The package lives under `src/`, so the editable install points there rather than at the repo
root:

```bash
git clone <your-repo-url>
cd FuncyFlows
python -m pip install -e "src[ot,examples,test]"
pytest src/tests
```

## Running the notebooks

They're checked in without outputs, deliberately — you're meant to run them, and it keeps the
repo from bloating with saved figures.

```bash
python -m pip install -e "src[examples]" jupyter
jupyter lab docs/notebooks
```

Each one finishes in a minute or two on a CPU. If one is taking ten, something's wrong.

## Building these docs

```bash
python -m pip install -r docs/requirements.txt
cd docs && make html          # -> docs/_build/html/index.html
```

:::{warning}
Use `python -m pip`, not bare `pip`, and check where things landed:

```bash
python -c "import sys, sphinx; print(sys.executable); print(sphinx.__file__)"
```

If you have both a venv and Anaconda on PATH — and on a Mac you probably do — the bare
`sphinx-build` script will happily grab the Anaconda one, which can't see the torch in your
venv. Autodoc imports the package, so that build fails in a confusing way. The Makefile calls
`python -m sphinx` for exactly this reason, but the install has to go to the right place too.
:::

The notebooks aren't executed during the build (`nb_execution_mode = "off"` in `conf.py`), so
you don't need torch for the prose pages. You do for the API pages, since autodoc imports
everything.

## The real examples

Bigger, slower, actually interesting:

```bash
python -m FuncyFlows.examples     # drops the scripts in your current directory
python bimodal_posterior.py
```
