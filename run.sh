# git pull

# Delete all `.DS_Store`
find . -name ".DS_Store" -delete

# De;ete all `.ipynb_checkpoints` folders
find . -name .ipynb_checkpoints -type d -exec rm -rf {} \;

# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute --inplace *.ipynb
# ls *.py|xargs -n 1 -P 4 ipython
# jupyter nbconvert --to script *.ipynb

ipython Colombia_Data.py
ipython Colombia_R_t.py

black *.py

git add --all && git commit -m "Update" && git push
