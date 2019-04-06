# ATIS-INTENT
Intent classification using NER BIOS tagging and NB

To install requirements 
pip install -r requirements.txt

cd /ATIS-INTENT

nohup python ATIS_INTENT.py --flag test &
--flag : indicates whether you want to train or test on testing data
if --flag test --> Test accuracy will be printed
if --flag train --> Training will happen

tail -f logs_info.log
