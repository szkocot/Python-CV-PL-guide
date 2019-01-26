#!/bin/bash

source activate cvguide

mkdir -p pdf
mkdir -p latex
mkdir -p docs
cd notebooks

for f in *.ipynb
do
    jupyter nbconvert --to markdown $f --output-dir ../docs
    jupyter nbconvert --to latex $f --output-dir ../latex --config convert_config.py
done

cd ../latex
for f in *.tex
do
    pdflatex $f -output-format pdf
done

mv *pdf ../pdf
cd ..
rm -rf latex