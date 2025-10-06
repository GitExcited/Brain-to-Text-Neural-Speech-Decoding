# Makefile for LaTeX with separate output dirs
MAIN=src/main
OUTDIR=out
PDFDIR=pdf

TEX=pdflatex
BIB=bibtex
TEXFLAGS=-interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(OUTDIR)

# Ensure LaTeX/BibTeX can find .sty, .bst, and .bib in src/
export TEXINPUTS := src:$(TEXINPUTS)
export BIBINPUTS := src:$(BIBINPUTS)
export BSTINPUTS := src:$(BSTINPUTS)

all: $(PDFDIR)/main.pdf

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(PDFDIR):
	mkdir -p $(PDFDIR)

# Build rule
$(PDFDIR)/main.pdf: $(MAIN).tex src/references.bib | $(OUTDIR) $(PDFDIR)
	$(TEX) $(TEXFLAGS) $(MAIN).tex
	$(BIB) $(OUTDIR)/main
	$(TEX) $(TEXFLAGS) $(MAIN).tex
	$(TEX) $(TEXFLAGS) $(MAIN).tex
	cp $(OUTDIR)/main.pdf $(PDFDIR)/

clean:
	rm -rf $(OUTDIR)/*

distclean: clean
	rm -rf $(PDFDIR)/*
