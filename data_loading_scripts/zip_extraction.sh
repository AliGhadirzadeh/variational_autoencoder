#!/bin/bash
unzip EEGdata.zip -x / -d ./subjects
unzip ./subjects/EEGdata.zip -x / -d ./subjects
rm ./subjects/EEGdata.zip
pwd
# transpose raw data
for filename in ./subjects/*; do
	echo "  transposing and converting: ${filename}"

	# add time header
	sed -i '1s/^/time/' ${filename}

	# transpose
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

mkdir ./subjects/npy
mv ./subjects/*.npy ./subjects/npy/

mkdir ./subjects/transposed
mv ./subjects/*_transpose ./subjects/transposed/

mkdir ./subjects/raw
mv ./subjects/? ./subjects/raw/
mv ./subjects/?? ./subjects/raw/