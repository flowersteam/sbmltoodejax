#!/bin/bash

# Generate PDF from markdown
pandoc paper.md --citeproc --lua-filter=scholarly-metadata.lua --lua-filter=author-info-blocks.lua --pdf-engine=xelatex -o paper.pdf
