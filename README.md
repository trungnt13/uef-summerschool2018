# UEF - Summer School 2018

[Programme and schedule](https://vvestman.github.io/summerschool/)

How to clone all the code and data provided to your computer (alternatively download .zip file from github page):

```bash
git clone --recursive https://github.com/trungnt13/uef-summerschool2018.git
```
For Windows users, using github desktop may significantly simplify the process:
[https://desktop.github.com/](https://desktop.github.com/)

## Setting up python environment

#### Installing miniconda
Following the instruction and install Miniconda from this link:
[https://conda.io/miniconda.html](https://conda.io/miniconda.html)

#### On Windows: Launch "Anaconda Prompt".

#### Create the environment
> conda env create -f=environment.yml

#### Using installed environment
Activating and using our environment:

For WINDOW:
> activate uefsummer18

For LINUX:
> source activate uefsummer18

Deactivating environment:
> source deactivate

Listing installed packages:
> conda list

#### Delete environment
> conda remove --name uefsummer18 --all

#### More tutorials for Windows users
[https://conda.io/docs/user-guide/install/windows.html#install-win-silent](https://conda.io/docs/user-guide/install/windows.html#install-win-silent)
