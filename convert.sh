#!/bin/bash

source activate cvguide
rm -rf build
rm -rf docs
mkdir build
mkdir docs
cd notebooks

for f in *.ipynb
do
    jupyter nbconvert --to markdown $f --output-dir ../docs
done

cd ../docs
dirlist=`ls ${prefix}*.md` 

pandoc $dirlist --latex-engine=xelatex --variable fontsize=10pt \
             --variable documentclass=scrbook --variable lang=polish \
             -H ../final.tex  -o ../build/book.pdf -f commonmark
pandoc $dirlist -o ../build/book.epub