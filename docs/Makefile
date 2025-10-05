# Makefile for LaTeX without latexmk
MAIN=src/main
PDF=$(MAIN:.tex=.pdf)

TEX=pdflatex
BIB=bibtex
TEXFLAGS=-interaction=nonstopmode -halt-on-error -file-line-error

all: $(PDF)

# note: depend on references.bib, not main.bib
$(PDF): $(MAIN).tex src/references.bib
	$(TEX) $(TEXFLAGS) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(TEXFLAGS) $(MAIN).tex
	$(TEX) $(TEXFLAGS) $(MAIN).tex

clean:
	rm -f src/*.aux src/*.log src/*.bbl src/*.blg src/*.out src/*.toc

distclean: clean
	rm -f $(PDF)
