#!/bin/bash

# Exctract and create directory structure
mkdir data
unzip ./eeg_data.zip -x
cp -r ./eeg_data/times ./data/
cp -r ./eeg_data/subs ./data/
rm -r ./eeg_data/
mkdir ./data/snippets

# Transpose raw data and convert to npy, including unit conversion
for filename in ./data/subs/*; do
   echo "Transposing and converting: ${filename}"

   # Add time header
   sed -i '1s/^/time/' ${filename}

   # Transpose
   awk '{
         for (f = 1; f <= NF; f++)
         a[NR, f] = $f
         }
         NF > nf { nf = NF }
         END {
         for (f = 1; f <= nf; f++)
            for (r = 1; r <= NR; r++)
               printf a[r, f] (r==NR ? RS : FS)
         }' ${filename} > ${filename}_transpose
done

rm ./data/subs/?
rm ./data/subs/??

zip -r data.zip data