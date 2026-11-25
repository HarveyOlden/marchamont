# Marchamont

Modern English → the English of a chosen year (1380–1790) or Scots (1500–1905). Runs entirely in the browser: text never leaves the page.

## Publish

1. Push this folder to a GitHub repository.
2. Settings → Pages → *Deploy from a branch* → `main` / root.
3. Edit `config.js` so `source` and `weights` point at your repository.

## Add the weights

1. Put the checkpoint in `pipeline/weights/final.pt` (ignored by git).
2. Export (any machine with Python, or `pipeline/export_colab.ipynb` on Colab):
   ```
   pip install -r pipeline/requirements.txt
   python pipeline/export_onnx.py
   ```
   This writes `model/meta.json` and the graph parts (each under 24 MB, so they upload through the GitHub web interface), and prints a check against the PyTorch model.
3. Commit everything in `model/`.

Re-exporting changes `meta.json`'s build date, which makes returning visitors fetch the new weights.

## Credits

Woodcuts: Museum Plantin-Moretus, Antwerp. Type: IM Fell by Igino Marini (SIL Open Font License, `fonts/OFL.txt`). Inference: ONNX Runtime Web (MIT, `vendor/ORT-LICENSE`).
