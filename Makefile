init:
	pip install -r requirements.txt
	git config --global user.email "smpsmp13@gmail.com"
	git config --global user.name "Soumya"


trainsm:
	/bin/python train.py

trainl:
	/bin/python train_lightning.py
