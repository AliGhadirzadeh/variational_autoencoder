#!/bin/bash

# Exctract and create directory structure
mkdir data
unzip ./eeg_data.zip -x
cp -r ./eeg_data/times ./data/
cp -r ./eeg_data/subs ./data/
rm -r ./eeg_data/
mkdir ./data/subs_npy
mkdir ./data/times_npy

# Transpose raw data and convert to npy, including unit conversion
for filename in ./data/subs/*; do
   echo "  transposing and converting: ${filename}"

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
      python3 ./txt2npy.py ${filename}
done

# Move to subs directory
mv ./data/subs/*.npy ./data/subs_npy/
rm -r ./data/subs

# Convert time to npy, including unit conversion
for filename in ./data/times/*; do
	python3 ./csv2npy.py ${filename}
done

# Move to times directory
mv ./data/times/*.npy ./data/times_npy/
rm -r ./data/times