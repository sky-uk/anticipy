SHELL := ${shell which bash}
SPHINXBUILD := ${shell which sphinx-build}

BUILDDIR = build
SOURCE = source
ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees $(SOURCE)

CLEAN = 	$(BUILDDIR)

all: html

check: 
ifeq (,${SPHINXBUILD})
	$(error sphinx-build not found. Try: pip install sphinx)
endif

html: check
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

shtml: check
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml

clean:
	rm -rf $(CLEAN)

help:
	@echo " "
	@echo " Use one of the following:"
	@echo "   help       - this message"
	@echo "   html       - everything in html"
	@echo "   singlehtml - everything in a single html"
	@echo "   clean      - removes all files in '$(BUILDDIR)' dir"
	@echo " "
