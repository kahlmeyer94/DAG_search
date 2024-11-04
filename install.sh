#install dag search

# remove directory if it exists
if [ -d "UDFS" ]; then
    rm -rf UDFS
fi

git clone https://github.com/kahlmeyer94/UDFS

cd UDFS
pip install -r requirements.txt
python setup.py install
