# ATIS-INTENT
Intent classification using NER BIOS tagging and NB

To install requirements:

conda create -n ATIS_INTENT python=3.7

pip install -r requirements.txt

For training or testing:

cd /ATIS-INTENT

conda activate ATIS_INTENT


nohup python ATIS_INTENT.py --flag test &

--flag : indicates whether you want to train or test on testing data

if --flag test : Test accuracy will be printed

if --flag train : Training will happen

tail -f logs_info.log
