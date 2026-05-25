# Running the pipeline locally with OpenAI gpt-image-1

The default pipeline (`template_extract_v3.py`) uses LaMa, a local open-source
inpainter that works offline but produces visible artefacts on complex
patterns. OpenAI's `gpt-image-1` (the model behind ChatGPT's image edit)
gives far better results on the same inputs — but it requires an API key
and reaches out to `api.openai.com`, which is **blocked from the remote
sandbox** the assistant is running in. So this is a workflow you run on
your own machine.

## One-time setup on your machine

```bash
git clone https://github.com/angadkohli/ask.git
cd ask/mockup_extractor

# Python 3.10+ recommended
pip install -r requirements.txt
pip install openai   # also already pinned in requirements.txt

export OPENAI_API_KEY=sk-...           # macOS / Linux
# or on Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

Put your template PSD(s) under `psd_samples/` and your mockup images under
any folder, e.g. `samples/`.

## Single run with OpenAI (best quality)

```bash
python template_extract_v3.py \
  --template psd_samples/cabin-suitcase-300dpi.psd \
  --mockup samples/IMG_5249.png \
  --out runs/openai \
  --print-sizes "16x20,18x24,22x29,24x32.125" \
  --use-openai --openai-quality high
```

## Batch

`batch_recover.py` doesn't yet pass `--use-openai` through; for now wrap the
single-mockup form in a shell loop:

```bash
for m in samples/*.png samples/*.jpg; do
  python template_extract_v3.py \
    --template psd_samples/cabin-suitcase-300dpi.psd \
    --mockup "$m" \
    --out "runs/openai/$(basename "${m%.*}")" \
    --print-sizes "16x20,18x24,22x29,24x32.125" \
    --use-openai --openai-quality high
done
```

## What gets sent to OpenAI

Per mockup the pipeline makes one API call per masked region, not one per
mockup:

  - Hardware regions (corner caps / wheels / zipper) — 1 call (one connected
    region; ~$0.05–0.19 at `high`).
  - Each non-band text region — 1 call each (typically 1–3 small text bits).
  - Each outpaint border at each requested print size — 1 call per print size
    (4 calls for the 4 default sizes).

So a typical Kiyara mockup with text-removal + 4 print sizes ≈ 8–10 calls.
At `gpt-image-1` high quality that's ~$1–$2/mockup. Drop to `medium`
(`--openai-quality medium`) to halve that.

## Quality tips

- `--openai-quality high` is the right setting for production prints.
- The pipeline still does the **lossless mask-and-lift** first; OpenAI only
  touches the hardware-occluded pixels and the outpaint borders. The 95%+
  of the design under transparent template pixels is byte-perfect from your
  mockup — no AI ever runs on those.
- If you don't want to spend on text-removal calls, run with text intact
  (`--no-remove-text`) and OpenAI is only used for hardware + outpaint.

## When OpenAI is genuinely worse

It can sometimes invent extra design elements when extending into a wider
aspect ratio (24×32 from a 0.62-aspect source). If you see hallucinated
flowers / leaves / etc. on the outpainted strips, fall back to reflect-only
extension by editing `export_at_print_size`'s `background_extend="reflect"`.
For truly perfect output at unusual aspect ratios there is no substitute
for original master art from the designer at that size.
