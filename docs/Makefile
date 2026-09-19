# Minimal Sphinx makefile.
#
# Sphinx is invoked as `python -m sphinx` rather than via the `sphinx-build` script, so the build
# always uses the interpreter that is active right now. The script form picks whatever comes first
# on PATH, which on a machine with both a venv and Anaconda is often not the one holding torch --
# and autodoc imports FuncyFlows, so it needs the one that does.
PYTHON        ?= python
SPHINXOPTS    ?=
SPHINXBUILD   ?= $(PYTHON) -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
