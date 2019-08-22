This directory contains files and scripts to extract the EEG data for the project.
eeg_data.zip contains three files:
	* subs - a directory containing per-subject EEG data
	* times - a directory containing per-subject csv files with times of discontinuities
	* SubWithoutSelection.txt - a file containing information about subjects without discontinuities
txt2npy.py is a help script to convert raw EEG-data in text format to npy format.
csv2npy.py is a helt script to convert discontinuity time information in csv format to npy format.
snippets_script.py is a help script to create continuous snips of EEG data given the raw EEG data and the discontinuity time information. Takes snippet length and window length as arguments

To extract data:
    * Run "sh data_extraction.sh"
    * Inside a directory containing the zipped data as "eeg_data.zip"
    * A folder "data" is then created, containing the data in different formats.
