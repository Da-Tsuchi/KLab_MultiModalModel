import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import contextlib
import io
import sys

class AccCounter():
    def __init__(self):
        self.accs = {'train':[], 'val':[]}

    def add(self, phase, loss):
        self.accs[phase].append(loss)

    def plot_loss(self, result_dir):
        # Plot the loss values.
        fig = plt.figure()
        plt.plot(self.accs['train'], label='Train')
        plt.plot(self.accs['val'], label='Val')

        # Set the title and axis labels.
        plt.title('Acvc Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()

        # Show the plot.
        plt.savefig(os.path.join(result_dir, "acc.png"))

class CiderCounter():
    def __init__(self):
        self.accs = {'train':[], 'val':[]}

    def add(self, phase, loss):
        self.accs[phase].append(loss)

    def plot_loss(self, result_dir):
        # Plot the loss values.
        fig = plt.figure()
        plt.plot(self.accs['train'], label='Train')
        plt.plot(self.accs['val'], label='Val')

        # Set the title and axis labels.
        plt.title('CIDEr Curve')
        plt.xlabel('Epoch')
        plt.ylabel('CIDEr')
        plt.legend()

        # Show the plot.
        plt.savefig(os.path.join(result_dir, "cider.png"))

class BleuCounter():
    def __init__(self):
        self.accs = {'train':[], 'val':[]}

    def add(self, phase, loss):
        self.accs[phase].append(loss)

    def plot_loss(self, result_dir):
        # Plot the loss values.
        fig = plt.figure()
        plt.plot(self.accs['train'], label='Train')
        plt.plot(self.accs['val'], label='Val')

        # Set the title and axis labels.
        plt.title('BLEU Curve')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU')
        plt.legend()

        # Show the plot.
        plt.savefig(os.path.join(result_dir, "bleu.png"))
        
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout context manager."""
    current_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Temporary redirect stdout to a fake stream
    try:
        yield
    finally:
        sys.stdout = current_stdout

def evaluate_score(pred_text, actual_text):
    """
    Compute CIDEr score using the Cider class from pycocoevalcap.
    
    Args:
        pred_text (list of str): List of predicted captions.
        actual_text (list of str): List of actual captions.
        
    Returns:
        float: CIDEr score.
    """
    # Ensure the predicted captions and actual captions are lists
    if not isinstance(pred_text, list):
        pred_text = [pred_text]
    if not isinstance(actual_text, list):
        actual_text = [actual_text]

    # Format predictions and actuals for CIDEr evaluation
    gts = {}
    res = {}
    for i, (pred, actual) in enumerate(zip(pred_text, actual_text)):
        # 文字列に変換
        pred = str(pred)
        actual = str(actual)
        gts[i] = [actual]
        res[i] = [pred]

    # Instantiate the CIDEr evaluator object
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr")
    ]
    print(gts)
    print(res)
    final_scores = {}
    
    with suppress_stdout():
        for scorer, metric in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(score) == list:
                for m, s in zip(metric, score):
                    final_scores[m] = s
            else:
                final_scores[metric] = score

    print(final_scores)
    return final_scores['CIDEr'], final_scores['Bleu_4']
    
# pred=["This is a cat","aiueo is a cat"]
# actual=["This is a dog","aiueo is a cat"]
# print(evaluate_score(pred, actual))
