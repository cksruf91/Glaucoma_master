#!/usr/bin/env sh

ls Training400/Non-Glaucoma/ | shuf | head -110 > temp.txt

while read -r line do
  mv Training400/Non-Glaucoma/"$line" Test/Non-Glaucoma/;
done<temp.txt

ls Training400/Glaucoma/ | shuf | head -12 > temp.txt

while read -r line do
  mv Training400/Glaucoma/"$line" Test/Glaucoma/;
done<temp.txt

rm -f temp.txt
