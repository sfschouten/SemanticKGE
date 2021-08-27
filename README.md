# SemanticKGE
LibKGE plugin implementing methods aimed at the incorporation of semantic information in knowledge graph embeddings.


## Installation
Make sure [LibKGE](https://github.com/uma-pi1/kge) is installed (see the repo for instructions).

Then, using pip:
```shell
git clone https://github.com/sfschouten/semantic-kge.git
pip install -e semantic-kge
```

Once SemKGE and LibKGE are both installed please follow LibKGE's instructions on downloading/preparing the data. 

After, one final data preprocessing step is necessary, which adds the type information to the datasets. This can be done with the following snippet of shell commands.

    cp -rT '<KGE_HOME>/data/wnrr' '<SEMKGE_HOME>/data/wnrr-typed'
	cp -rT '<KGE_HOME>/data/fb15k-237' '<SEMKGE_HOME>/data/fb15k-237-typed'

	cd <SEMKGE_HOME>/data
    mkdir -p ".tmp"
    
	cd <SEMKGE_HOME>/data/.tmp
	!curl -L https://surfdrive.surf.nl/files/index.php/s/N1c8VRH0I6jTJuN/download --output wn18rr.tar.gz
	!curl -L https://surfdrive.surf.nl/files/index.php/s/rGqLTDXRFLPJYg7/download --output fb15k-237.tar.gz
	tar xzf wn18rr.tar.gz
	tar xzf fb15k-237.tar.gz
	cp WN18RR/entity2type.txt <SEMKGE_HOME>/data/wnrr-typed/entity_types.txt
	cp FB15k-237/entity2type.txt <SEMKGE_HOME>/data/fb15k-237-typed/entity_types.txt
	cd ..
	
	python preprocess_types.py wnrr-typed
	python preprocess_types.py fb15k-237-typed


## Results - Existing Methods
|        | MRR | Hits@1 | Hits@3 | Hits@10 | Config file |
|--------|-----|--------|--------|---------|-------------|
| TransT |     |        |        |         |             |
|        |     |        |        |         |             |
|        |     |        |        |         |             |

## Results - Our Methods
|        | MRR | Hits@1 | Hits@3 | Hits@10 | Config file |
|--------|-----|--------|--------|---------|-------------|
|        |     |        |        |         |             |
|        |     |        |        |         |             |
|        |     |        |        |         |             |
