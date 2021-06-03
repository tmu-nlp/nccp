# Neural Combinatory Constituency Parsers

![NCCP](000/figures/nccp.gif)

## Requirements
- `pip install -r requirements/visual.txt` to visualize remote tensors locally through sftp. (funny!)
- `pip install -r requirements/full.txt` to train or test our models with PyTorch.
- [Evalb](https://nlp.cs.nyu.edu/evalb/) and [fastText](https://fasttext.cc/).
  - You need to download and configure `manager.yaml` for them.
- [Huggingface transformers](https://github.com/huggingface/transformers).

## Models
- NCCP: Neural Combinatory Constituency Parsing (binary)
  - ACCP: Attentional Combinatory Constituency Parsing (multi-branching)

## Usage

### Visualization
To see training process of English parsing on a free PTB devel set (section 00),
- `./visualization.py '000/lstm_nccp/0.stratified.ptb.L85R15/ptb_devel'` (binary, empirical linear compleixty)

To see English parsing on a free PTB test set (section 01),
- `./visualization.py '000/lstm_nccp/1.triangular.ptb.L85R15/ptb_devel'` (binary, squared complexity).

For multi-branching tree visualization with headedness (section 01),
- Check `'000/lstm_accp/0.multi.branching.ptb/ptb_devel/*.art'` text files.

(We use freely available Penn Treebank/PTB sections for this visualization.)

If you want to see more, please download our pre-trained models in [000.zip](http://cl.sd.tmu.ac.jp/~zchen/000.zip).

You may run our pre-trained models on your test sets.
Or, you may also train a new model with your own corpus to see more details.
To do that, you need to prepare data by setting a configuration, e.g., `000/manager.yaml`.

### Test a Pre-Trained Models
- Set key `data:ptb:source_path:` in `000/manager.yaml` to your PTB/WSJ folder, and then use `./manager.py 000 -p` to prepare the data.
- Set keys under `tool:evalb:` for F1 scores. (Use `./manager.py 000` to check the status. Any improper configuration will be prompted.)
- Use `./manager.py 000 -s lstm_nccp -i0` to test the pre-train binary model on stratified PTB test set.
  - Add `-g [GPU ID]` to choose a GPU; the default is 0.
  - Use `./visualization.py '000/lstm_nccp/0.stratified.L85R15.ptb/ptb_test'` to see local visualization or add ` -h [server address] -p [port]` to see remote one.

### Train a New Model
If you want a new work folder, try `mkdir Oh-My-Folder; ./manager.py Oh-My-Folder`. You will get a new configure file `Oh-My-Folder/manager.yaml`.
- Use `./manager.py Oh-My-Folder` to check the status and available experiments.
- Use `./manager.py Oh-My-Folder -s lstm_nccp/ptb:Oh-My-Model` to train a continuous model on PTB.
  - Add `-x [fv=fine evaluation start:[early stop count:[fine eval count]]],[max=max epoch count],[! test along with evaluation]` to change training settings.
  - Use `-s [[lstm_nccp|lstm_accp]/[ptb|ctb|ktb]]:My-Suffix` to choose a continuous parsing experiment.
  - Use `-s [xlnet_nccp|xlnet_accp]:My-Suffix` to involve a pre-trained XLNet model.

### Tips
- Try modifying the hyper-parameters in your `Oh-My-Folder/manager.yaml`.
- `Oh-My-Folder/lstm_nccp/register_and_tests.yaml` contains the scores for this experiment.
- Our reported top speed of NCCP is 1.3k sents/sec on GeForce GTX 1080 Ti with `task:[select]:train:multiprocessing_decode: true`. The speed is shown during the training process. To test the speed on the test set. Use `-x !` to include test set with evaluation. Otherwise, the top speed is around 0.5k sents/sec.

### Tips for Researchers and Developers
- All modules in `experiments` prefixed with `t_` will be recognized by `manager.py`.
- `models/` contains our models' base classes such as `nccp.py` and `accp.py`.
  If you are looking for our implementation, these files plus `combine.py` are what you are looking for.
This framework is mainly for research. We do not pack vocabulary to the model.
The models do not stand along outside its original folder.
They may produce a lot of bug if you go outside of the dataset splits.

- [Tokyo Metropolitan University - NLP Group (Komachi Lab)](http://cl.sd.tmu.ac.jp/en/)
- Contact me: feelmailczs[at]gmail.com
