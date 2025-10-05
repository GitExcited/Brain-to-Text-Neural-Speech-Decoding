FROM mcr.microsoft.com/devcontainers/universal:noble
WORKDIR /workspace
COPY . /workspace
RUN apt-get update && \
    apt-get install -y \
        texlive-base \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-science \
        texlive-bibtex-extra && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    ls