# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	rm -rf $(BUILDDIR)/
	rm -rf source/_build/
	rm -rf source/_generated/
	rm -rf source/test-report/_generated/

clean_api:
	rm -rf source/_generated/
	rm -rf source/test-report/_generated/

html_api:
	make clean_api
	cd .. ; python setup.py build_ext -i -Oprofile -j8
	make html

all:
	make html
	# make latexpdf
	make deploy

deploy:
	rm -rf ../choreo-docs/*
	cp -r build/html/* ../choreo-docs/
	touch ../choreo-docs/.nojekyll

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -j auto

