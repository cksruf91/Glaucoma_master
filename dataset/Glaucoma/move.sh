# bash

ls Training400/Non-Glaucoma/ | shuf | head -110 > temp.txt

files=`cat temp.txt`
for line in $files;
do
       mv Training400/Non-Glaucoma/$line Test/Non-Glaucoma/
done

rm -f temp.txt


ls Training400/Glaucoma/ | shuf | head -12 > temp.txt

files=`cat temp.txt`
for line in $files;
do
        mv Training400/Glaucoma/$line Test/Glaucoma/
done

rm -f temp.txt
