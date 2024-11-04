#install dag search

# remove directory if it exists
if [ -d "Unbiased-DAG-Frame-Search" ]; then
    rm -rf Unbiased-DAG-Frame-Search
fi

git clone https://github.com/kahlmeyer94/Unbiased-DAG-Frame-Search

cd Unbiased-DAG-Frame-Search
pip install -r requirements.txt
python setup.py install
