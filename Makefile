gitconfig:
	git config --global user.email "smpsmp13@gmail.com"
	git config --global user.name "Soumya"

install-screen:
	apt update
	apt install screen -y

init: gitconfig install-screen
	pip install -r requirements.txt

trainsm:
	/bin/python train.py

trainl:
	/bin/python train_lightning.py
