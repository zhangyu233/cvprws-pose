"""Trust evaluation (placeholder).

Planned metrics:
- Joint failure detection: AUROC / AUPR with score = 1 - t_j
- Risk-coverage curve + AURC
- Calibration: ECE

Implementation suggestion:
- Use `mim test ... --out results.pkl` (or json)
- Compare predicted joints with COCO GT joints (normalized by bbox area)
"""


def main():
    raise NotImplementedError('TODO: implement trust evaluation.')


if __name__ == '__main__':
    main()
