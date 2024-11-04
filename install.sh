#install dag search

# remove directory if it exists
if [ -d "DAG-Search" ]; then
    rm -rf DAG-Search
fi

git clone https://github.com/kahlmeyer94/DAG-Search

cd DAG-Search
pip install -r requirements.txt
python setup.py install
